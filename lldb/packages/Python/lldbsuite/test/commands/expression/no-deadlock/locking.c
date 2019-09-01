#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

pthread_mutex_t contended_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t control_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  control_condition;

pthread_mutex_t thread_started_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  thread_started_condition;

// This function runs in a thread.  The locking dance is to make sure that 
// by the time the main thread reaches the pthread_join below, this thread
// has for sure acquired the contended_mutex.  So then the call_me_to_get_lock
// function will block trying to get the mutex, and only succeed once it
// signals this thread, then lets it run to wake up from the cond_wait and
// release the mutex.

void *
lock_acquirer_1 (void *input)
{
  pthread_mutex_lock (&contended_mutex);
  
  // Grab this mutex, that will ensure that the main thread
  // is in its cond_wait for it (since that's when it drops the mutex.

  pthread_mutex_lock (&thread_started_mutex);
  pthread_mutex_unlock(&thread_started_mutex);

  // Now signal the main thread that it can continue, we have the contended lock
  // so the call to call_me_to_get_lock won't make any progress till  this
  // thread gets a chance to run.

  pthread_mutex_lock (&control_mutex);

  pthread_cond_signal (&thread_started_condition);

  pthread_cond_wait (&control_condition, &control_mutex);

  pthread_mutex_unlock (&contended_mutex);
  return NULL;
}

int
call_me_to_get_lock ()
{
  pthread_cond_signal (&control_condition);
  pthread_mutex_lock (&contended_mutex);
  return 567;
}

int main ()
{
  pthread_t thread_1;

  pthread_cond_init (&control_condition, NULL);
  pthread_cond_init (&thread_started_condition, NULL);

  pthread_mutex_lock (&thread_started_mutex);

  pthread_create (&thread_1, NULL, lock_acquirer_1, NULL);
  
  pthread_cond_wait (&thread_started_condition, &thread_started_mutex);

  pthread_mutex_lock (&control_mutex);
  pthread_mutex_unlock (&control_mutex);

  // Break here.  At this point the other thread will have the contended_mutex,
  // and be sitting in its cond_wait for the control condition.  So there is
  // no way that our by-hand calling of call_me_to_get_lock will proceed
  // without running the first thread at least somewhat.

  call_me_to_get_lock();
  pthread_join (thread_1, NULL);

  return 0;

}
