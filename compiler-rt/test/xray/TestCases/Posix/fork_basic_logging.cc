// Check that when forking in basic logging mode, we get the different tids for child and parent
// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm -f fork-basic-logging-test-*
// RUN: XRAY_OPTIONS="patch_premain=true xray_logfile_base=fork-basic-logging-test- \
// RUN:     xray_mode=xray-basic verbosity=1 xray_naive_log_func_duration_threshold_us=0" \
// RUN:     %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:     "`ls -S fork-basic-logging-test-* | head -1`" \
// RUN:     | FileCheck %s --check-prefix=TRACE

// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree

// Not ported.
// UNSUPPORTED: netbsd

#include "xray/xray_log_interface.h"
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/syscall.h>

//modified from sanitizer

static uintptr_t syscall_gettid() {
  uint64_t retval;
  asm volatile("syscall" : "=a"(retval) : "a"(__NR_gettid) : "rcx", "r11",
               "memory", "cc");
  return retval;
}

/////////////

static uint64_t parent_tid;

[[clang::xray_always_instrument]]
uint64_t __attribute__((noinline)) log_syscall_gettid()
{
	//don't optimize this function away
	uint64_t tid = syscall_gettid();
	printf("Logging tid %lu\n", tid);
	return tid;
}

[[clang::xray_always_instrument, clang::xray_log_args(1)]]
void __attribute__((noinline)) print_parent_tid(uint64_t tid)
{
	printf("Parent with tid %lu", tid);
}

[[clang::xray_always_instrument, clang::xray_log_args(1)]]
void __attribute__((noinline)) print_child_tid(uint64_t tid)
{
	printf("Child with tid %lu", tid);
}

[[clang::xray_always_instrument]] void __attribute__((noinline)) print_parent_or_child()
{
	uint64_t tid = syscall_gettid();
	if(tid == parent_tid)
	{
		print_parent_tid(tid);
	}
	else
	{
		print_child_tid(tid);
	}
}

int main()
{
	parent_tid = log_syscall_gettid();
	if(fork())
	{
		print_parent_or_child();
  		// CHECK-DAG: Parent with tid
	}
	else
	{
		print_parent_or_child();
  		// CHECK-DAG: Child with tid
	}
	return 0;
}

// Make sure we know which thread is the parent process
// TRACE-DAG: - { type: 0, func-id: [[LSGT:[0-9]+]], function: {{.*log_syscall_gettid.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], process: [[PROCESS1:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }

// TRACE-DAG: - { type: 0, func-id: [[PPOC:[0-9]+]], function: {{.*print_parent_or_child.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS1]], kind: function-enter, tsc: {{[0-9]+}} }
//
// The parent will print its pid
// TRACE-DAG: - { type: 0, func-id: [[PPTARG:[0-9]+]], function: {{.*print_parent_tid.*}}, args: [ [[THREAD1]] ], cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS1]], kind: function-enter-arg, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[PPTARG]], function: {{.*print_parent_tid.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS1]], kind: function-exit, tsc: {{[0-9]+}} }
//
// TRACE-DAG  - { type: 0, func-id: [[PPOC]], function: {{.*print_parent_or_child.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS1]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}} }

// TRACE-DAG: - { type: 0, func-id: [[PPOC]], function: {{.*print_parent_or_child.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], process: [[PROCESS2:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
//
// The child will print its pid
// TRACE-DAG: - { type: 0, func-id: [[PCTARG:[0-9]+]], function: {{.*print_child_tid.*}}, args: [ [[THREAD2]] ], cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS2]], kind: function-enter-arg, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[PCTARG]], function: {{.*print_child_tid.*}}, cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS2]], kind: function-exit, tsc: {{[0-9]+}} }
//
// TRACE-DAG: - { type: 0, func-id: [[PPOC]], function: {{.*print_parent_or_child.*}}, cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS2]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}} }
