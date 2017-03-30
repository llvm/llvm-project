//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which multiple events
// (breakpoints, watchpoints, crashes, and signal generation/delivery) happen
// from multiple threads. The test expects the debugger to set a breakpoint on
// the main thread (before any worker threads are spawned) and modify variables
// which control the number of threads that are spawned for each action.

#include "pseudo_barrier.h"
#include <vector>
using namespace std;

#include <pthread.h>

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

typedef std::vector<std::pair<unsigned, void*(*)(void*)> > action_counts;
typedef std::vector<pthread_t> thread_vector;

pseudo_barrier_t g_barrier;
int g_breakpoint = 0;
int g_sigusr1_count = 0;
std::atomic_int g_watchme;

struct action_args {
  int delay;
};

// Perform any extra actions required by thread 'input' arg
void do_action_args(void *input) {
    if (input) {
      action_args *args = static_cast<action_args*>(input);
      sleep(args->delay);
    }
}

void *
breakpoint_func (void *input)
{
    // Wait until all threads are running
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    // Do something
    g_breakpoint++;       // Set breakpoint here
    return 0;
}

void *
signal_func (void *input) {
    // Wait until all threads are running
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    // Send a user-defined signal to the current process
    //kill(getpid(), SIGUSR1);
    // Send a user-defined signal to the current thread
    pthread_kill(pthread_self(), SIGUSR1);

    return 0;
}

void *
watchpoint_func (void *input) {
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    g_watchme += 1;     // watchpoint triggers here
    return 0;
}

void *
crash_func (void *input) {
    pseudo_barrier_wait(g_barrier);
    do_action_args(input);

    int *a = 0;
    *a = 5; // crash happens here
    return 0;
}

void sigusr1_handler(int sig) {
    if (sig == SIGUSR1)
        g_sigusr1_count += 1; // Break here in signal handler
}

/// Register a simple function for to handle signal
void register_signal_handler(int signal, void (*handler)(int))
{
    sigset_t empty_sigset;
    sigemptyset(&empty_sigset);

    struct sigaction action;
    action.sa_sigaction = 0;
    action.sa_mask = empty_sigset;
    action.sa_flags = 0;
    action.sa_handler = handler;
    sigaction(SIGUSR1, &action, 0);
}

void start_threads(thread_vector& threads,
                   action_counts& actions,
                   void* args = 0) {
    action_counts::iterator b = actions.begin(), e = actions.end();
    for(action_counts::iterator i = b; i != e; ++i) {
        for(unsigned count = 0; count < i->first; ++count) {
            pthread_t t;
            pthread_create(&t, 0, i->second, args);
            threads.push_back(t);
        }
    }
}

int dotest()
{
    g_watchme = 0;

    // Actions are triggered immediately after the thread is spawned
    unsigned num_breakpoint_threads = 1;
    unsigned num_watchpoint_threads = 0;
    unsigned num_signal_threads = 1;
    unsigned num_crash_threads = 0;

    // Actions below are triggered after a 1-second delay
    unsigned num_delay_breakpoint_threads = 0;
    unsigned num_delay_watchpoint_threads = 0;
    unsigned num_delay_signal_threads = 0;
    unsigned num_delay_crash_threads = 0;

    register_signal_handler(SIGUSR1, sigusr1_handler); // Break here and adjust num_[breakpoint|watchpoint|signal|crash]_threads

    unsigned total_threads = num_breakpoint_threads \
                             + num_watchpoint_threads \
                             + num_signal_threads \
                             + num_crash_threads \
                             + num_delay_breakpoint_threads \
                             + num_delay_watchpoint_threads \
                             + num_delay_signal_threads \
                             + num_delay_crash_threads;

    // Don't let either thread do anything until they're both ready.
    pseudo_barrier_init(g_barrier, total_threads);

    action_counts actions;
    actions.push_back(std::make_pair(num_breakpoint_threads, breakpoint_func));
    actions.push_back(std::make_pair(num_watchpoint_threads, watchpoint_func));
    actions.push_back(std::make_pair(num_signal_threads, signal_func));
    actions.push_back(std::make_pair(num_crash_threads, crash_func));

    action_counts delay_actions;
    delay_actions.push_back(std::make_pair(num_delay_breakpoint_threads, breakpoint_func));
    delay_actions.push_back(std::make_pair(num_delay_watchpoint_threads, watchpoint_func));
    delay_actions.push_back(std::make_pair(num_delay_signal_threads, signal_func));
    delay_actions.push_back(std::make_pair(num_delay_crash_threads, crash_func));

    // Create threads that handle instant actions
    thread_vector threads;
    start_threads(threads, actions);

    // Create threads that handle delayed actions
    action_args delay_arg;
    delay_arg.delay = 1;
    start_threads(threads, delay_actions, &delay_arg);

    // Join all threads
    typedef std::vector<pthread_t>::iterator thread_iterator;
    for(thread_iterator t = threads.begin(); t != threads.end(); ++t)
        pthread_join(*t, 0);

    return 0;
}

int main ()
{
    dotest();
    return 0; // Break here and verify one thread is active.
}


