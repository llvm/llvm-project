// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -O2 %s -Fe%t.dll
// RUNX: %run %t %t.dll 2>&1 | FileCheck %s

// Check that ASan does not CHECK fail when SEH is used around a crash from a
// thread injected by control C.

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

static void __declspec(noinline) CrashOnProcessDetach() {
  printf("CrashOnProcessDetach\n");
  fflush(stdout);
  *static_cast<volatile int *>(0) = 0x356;
}

bool g_is_child = false;

BOOL WINAPI DllMain(PVOID h, DWORD reason, PVOID reserved) {
  if (reason == DLL_PROCESS_DETACH && g_is_child) {
    printf("in DllMain DLL_PROCESS_DETACH\n");
    fflush(stdout);
    __try {
      CrashOnProcessDetach();
    } __except (1) {
      printf("caught crash\n");
      fflush(stdout);
    }
  }
  return true;
}

static void run_child() {
  // Send this process group Ctrl+C. That should only be this process.
  printf("GenerateConsoleCtrlEvent\n");
  fflush(stdout);
  GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
  Sleep(10 * 1000); // Wait 10 seconds, and the process should die.
  printf("unexpected execution after interrupt\n");
  fflush(stdout);
  exit(0x42);
}

static int WINAPI ignore_control_c(DWORD ctrl_type) {
  // Don't interrupt the parent.
  return ctrl_type == CTRL_C_EVENT;
}

static int run_parent() {
  // Set an environment variable to tell the child process to interrupt itself.
  if (!SetEnvironmentVariableW(L"DO_CONTROL_C", L"1")) {
    printf("SetEnvironmentVariableW failed (0x%8lx).\n", GetLastError());
    fflush(stdout);
    return 2;
  }

  // Launch a new process using the current executable with a new console.
  // Ctrl-C events are console-wide, so we need a new console.
  STARTUPINFOW si;
  memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  // Hides the new console window that we are creating.
  si.dwFlags |= STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;
  // Ensures that stdout still goes to the parent despite the new console.
  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

  PROCESS_INFORMATION pi;
  memset(&pi, 0, sizeof(pi));
  int flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_NEW_CONSOLE;
  if (!CreateProcessW(nullptr,           // No module name (use command line)
                      GetCommandLineW(), // Command line
                      nullptr,           // Process handle not inheritable
                      nullptr,           // Thread handle not inheritable
                      TRUE,              // Set handle inheritance to TRUE
                      flags,             // Flags to give the child a console
                      nullptr,           // Use parent's environment block
                      nullptr,           // Use parent's starting directory
                      &si, &pi)) {
    printf("CreateProcess failed (0x%08lx).\n", GetLastError());
    fflush(stdout);
    return 2;
  }

  // Wait until child process exits.
  if (WaitForSingleObject(pi.hProcess, INFINITE) == WAIT_FAILED) {
    printf("WaitForSingleObject failed (0x%08lx).\n", GetLastError());
    fflush(stdout);
    return 2;
  }

  // Get the exit code. It should be the one for ctrl-c events.
  DWORD rc;
  if (!GetExitCodeProcess(pi.hProcess, &rc)) {
    printf("GetExitCodeProcess failed (0x%08lx).\n", GetLastError());
    fflush(stdout);
    return 2;
  }
  if (rc == STATUS_CONTROL_C_EXIT)
    printf("child quit with STATUS_CONTROL_C_EXIT\n");
  else
    printf("unexpected exit code: 0x%08lx\n", rc);
  fflush(stdout);

  // Close process and thread handles.
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return 0;
}

// CHECK: in DllMain DLL_PROCESS_DETACH
// CHECK: CrashOnProcessDetach
// CHECK: caught crash
// CHECK: child quit with STATUS_CONTROL_C_EXIT

extern "C" int __declspec(dllexport) test_function() {
  wchar_t buf[260];
  int len = GetEnvironmentVariableW(L"DO_CONTROL_C", buf, 260);
  if (len > 0) {
    g_is_child = true;
    run_child();
  } else {
    exit(run_parent());
  }
  return 0;
}
