/* This is a test case where the parent process forks 10 children
 * which contend to merge profile data to the same file. With
 * file locking support, the data from each child should not
 * be lost.
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

void spawn_child(PROCESS_INFORMATION *pi, int child_num) {
  wchar_t child_str[10];
  _itow(child_num, child_str, 10);
  if (!SetEnvironmentVariableW(L"CHILD_NUM", child_str)) {
    printf("SetEnvironmentVariableW failed (0x%8lx).\n", GetLastError());
    fflush(stdout);
    exit(1);
  }

  STARTUPINFOW si;
  memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);

  memset(pi, 0, sizeof(PROCESS_INFORMATION));

  if (!CreateProcessW(NULL,              // No module name (use command line)
                      GetCommandLineW(), // Command line
                      NULL,              // Process handle not inheritable
                      NULL,              // Thread handle not inheritable
                      TRUE,              // Set handle inheritance to TRUE
                      0,                 // No flags
                      NULL,              // Use parent's environment block
                      NULL,              // Use parent's starting directory
                      &si, pi)) {
    printf("CreateProcess failed (0x%08lx).\n", GetLastError());
    fflush(stdout);
    exit(1);
  }
}

int wait_child(PROCESS_INFORMATION *pi) {
  WaitForSingleObject(pi->hProcess, INFINITE);

  DWORD exit_code;
  if (!GetExitCodeProcess(pi->hProcess, &exit_code)) {
    printf("GetExitCodeProcess failed (0x%08lx).\n", GetLastError());
    fflush(stdout);
    exit(1);
  }

  CloseHandle(pi->hProcess);
  CloseHandle(pi->hThread);

  return exit_code;
}

#define NUM_CHILDREN 10

int foo(int num) {
  if (num < (NUM_CHILDREN / 2)) {
    return 1;
  } else if (num < NUM_CHILDREN) {
    return 2;
  }
  return 3;
}

int main(int argc, char *argv[]) {
  char *child_str = getenv("CHILD_NUM");
  if (!child_str) {
    PROCESS_INFORMATION child[NUM_CHILDREN];
    // In parent
    for (int i = 0; i < NUM_CHILDREN; i++) {
      spawn_child(&child[i], i);
    }
    for (int i = 0; i < NUM_CHILDREN; i++) {
      wait_child(&child[i]);
    }
    return 0;
  } else {
    // In child
    int child_num = atoi(child_str);
    int result = foo(child_num);
    if (result == 3) {
      fprintf(stderr, "Invalid child count!");
      return 1;
    }
    return 0;
  }
}
