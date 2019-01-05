#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <pthread.h>

int finished_enqueueing_work = 0;
atomic_int thread_count = 0;

void
doing_the_work_1(void *in)
{
    atomic_fetch_add(&thread_count, 1);
    while (1)
        sleep (1);
}

void
submit_work_1a(void *in)
{
    dispatch_queue_t *work_performer_1 = (dispatch_queue_t*) in;
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
}

void
submit_work_1b(void *in)
{
    dispatch_queue_t *work_performer_1 = (dispatch_queue_t*) in;
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
    dispatch_async_f (*work_performer_1, NULL, doing_the_work_1);
    while (1)
      sleep (1);
}

void
doing_the_work_2(void *in)
{
    while (1)
        sleep (1);
}

void
submit_work_2(void *in)
{
    dispatch_queue_t *work_performer_2 = (dispatch_queue_t*) in;
    int i = 0;
    while (i++ <  5000)
    {
        dispatch_async_f (*work_performer_2, NULL, doing_the_work_2);
        dispatch_async_f (*work_performer_2, NULL, doing_the_work_2);
    }
    finished_enqueueing_work = 1;
}


void
doing_the_work_3(void *in)
{
    while (1)
        sleep(1);
}

void
submit_work_3(void *in)
{
    dispatch_queue_t *work_performer_3 = (dispatch_queue_t*) in;
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
    dispatch_async_f (*work_performer_3, NULL, doing_the_work_3);
}


void
stopper ()
{
    while (1)
        sleep (1);
}


int main (int argc, const char **argv)
{
    dispatch_queue_t work_submittor_1 = dispatch_queue_create ("com.apple.work_submittor_1", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t work_submittor_2 = dispatch_queue_create ("com.apple.work_submittor_and_quit_2", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t work_submittor_3 = dispatch_queue_create ("com.apple.work_submittor_3", DISPATCH_QUEUE_SERIAL);

    dispatch_queue_t work_performer_1 = dispatch_queue_create ("com.apple.work_performer_1", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_t work_performer_2 = dispatch_queue_create ("com.apple.work_performer_2", DISPATCH_QUEUE_SERIAL);

    dispatch_queue_t work_performer_3 = dispatch_queue_create ("com.apple.work_performer_3", DISPATCH_QUEUE_CONCURRENT);

    dispatch_async_f (work_submittor_1, (void*) &work_performer_1, submit_work_1a);
    dispatch_async_f (work_submittor_1, (void*) &work_performer_1, submit_work_1b);

    dispatch_async_f (work_submittor_2, (void*) &work_performer_2, submit_work_2);

    dispatch_async_f (work_submittor_3, (void*) &work_performer_3, submit_work_3);


    // Spin up threads with each of the different libdispatch QoS values.
    dispatch_async (dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            pthread_setname_np ("user initiated QoS");
            atomic_fetch_add(&thread_count, 1);
            while (1)
                sleep (10);
                });
    dispatch_async (dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
            pthread_setname_np ("user interactive QoS");
            atomic_fetch_add(&thread_count, 1);
            while (1)
                sleep (10);
                });
    dispatch_async (dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
            pthread_setname_np ("default QoS");
            atomic_fetch_add(&thread_count, 1);
            while (1)
                sleep (10);
                });
    dispatch_async (dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            pthread_setname_np ("utility QoS");
            atomic_fetch_add(&thread_count, 1);
            while (1)
                sleep (10);
                });
    dispatch_async (dispatch_get_global_queue(QOS_CLASS_BACKGROUND, 0), ^{
            pthread_setname_np ("background QoS");
            atomic_fetch_add(&thread_count, 1);
            while (1)
                sleep (10);
                });
    dispatch_async (dispatch_get_global_queue(QOS_CLASS_UNSPECIFIED, 0), ^{
            pthread_setname_np ("unspecified QoS");
            atomic_fetch_add(&thread_count, 1);
            while (1)
                sleep (10);
                });

    // Unfortunately there is no pthread_barrier on darwin.
    while (atomic_load(&thread_count) < 7)
        sleep(1);

    while (finished_enqueueing_work == 0)
        sleep (1);
    stopper ();

}
