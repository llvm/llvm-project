// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

#include "callback.h"
#include <omp.h>

__attribute__ ((noinline)) // workaround for bug in icc
void print_task_type(int id)
{
  #pragma omp critical
  {
    int task_type;
    char buffer[2048];
    ompt_get_task_info(0, &task_type, NULL, NULL, NULL, NULL);
    format_task_type(task_type, buffer);
    printf("%" PRIu64 ": id=%d task_type=%s=%d\n", ompt_get_thread_data()->value, id, buffer, task_type);
  }
};

int main()
{
  //initial task
  print_task_type(0);

  int x;
  //implicit task
  #pragma omp parallel num_threads(1)
  {
    print_task_type(1);
    x++;
  }

  #pragma omp parallel num_threads(1)
  #pragma omp master
  {
    //explicit task
    #pragma omp task
    {
      print_task_type(2);
      x++;
    }

    //explicit task with undeferred
    #pragma omp task if(0)
    {
      print_task_type(3);
      x++;
    }

    //explicit task with untied
    #pragma omp task untied
    {
      print_task_type(4);
      x++;
    }

    //explicit task with final
    #pragma omp task final(1)
    {
      print_task_type(5);
      x++;
      //nested explicit task with final and undeferred
      #pragma omp task
      {
        print_task_type(6);
        x++;
      }
    }

/*
    //TODO:not working
    //explicit task with mergeable
    #pragma omp task mergeable
    {
      print_task_type(7);
      x++;
    }
*/

    //TODO: merged task
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, actual_parallelism=1, index=1, flags=1
  // CHECK: {{^}}[[MASTER_ID]]: id=0 task_type=ompt_task_initial=1
  // CHECK: {{^}}[[MASTER_ID]]: id=1 task_type=ompt_task_implicit|ompt_task_undeferred=134217730

  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred=134217732, has_dependences=no
  // CHECK: {{^[0-9]+}}: id=2 task_type=ompt_task_explicit|ompt_task_undeferred=134217732

  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred=134217732, has_dependences=no
  // CHECK: {{^[0-9]+}}: id=3 task_type=ompt_task_explicit|ompt_task_undeferred=134217732

  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_untied=402653188, has_dependences=no
  // CHECK: {{^[0-9]+}}: id=4 task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_untied=402653188

  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_final=671088644, has_dependences=no
  // CHECK: {{^[0-9]+}}: id=5 task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_final=671088644

  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_final=671088644, has_dependences=no
  // CHECK: {{^[0-9]+}}: id=6 task_type=ompt_task_explicit|ompt_task_undeferred|ompt_task_final=671088644

  // ___CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit|ompt_task_undeferred=134217732, has_dependences=no
  // ___CHECK: {{^[0-9]+}}: id=7 task_type=ompt_task_explicit|ompt_task_undeferred=134217732

  return 0;
}
