// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_task_final()
{
  int tids[NUM_TASKS];
  int includedtids[NUM_TASKS];
  int i;
  int error = 0;
  #pragma omp parallel
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        /* First we have to store the value of the loop index in a new variable
         * which will be private for each task because otherwise it will be overwritten
         * if the execution of the task takes longer than the time which is needed to
         * enter the next step of the loop!
         */
        int myi;
        myi = i;

        #pragma omp task final(i>=10)
        {
          tids[myi] = omp_get_thread_num();
          /* we generate included tasks for final tasks */
          if(myi >= 10) {
            int included = myi;
            #pragma omp task
            {
              my_sleep (SLEEPTIME);
              includedtids[included] = omp_get_thread_num();
            } /* end of omp included task of the final task */
            my_sleep (SLEEPTIME);
          } /* end of if it is a final task*/
        } /* end of omp task */
      } /* end of for */
    } /* end of single */
  } /*end of parallel */

  /* Now we ckeck if more than one thread executed the final task and its included task. */
  for (i = 10; i < NUM_TASKS; i++) {
    if (tids[i] != includedtids[i]) {
      error++;
    }
  }
  return (error==0);
} /* end of check_paralel_for_private */

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_final()) {
      num_failed++;
    }
  }
  return num_failed;
}

