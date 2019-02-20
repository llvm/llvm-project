//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

class Task {
public:
    int id;
    Task *next;
    Task(int i, Task *n):
        id(i),
        next(n)
    {}
};


int main (int argc, char const *argv[])
{
    Task *task_head = NULL;
    Task *task1 = new Task(1, NULL);
    Task *task2 = new Task(2, NULL);
    Task *task3 = new Task(3, NULL); // Orphaned.
    Task *task4 = new Task(4, NULL);
    Task *task5 = new Task(5, NULL);

    task_head = task1;
    task1->next = task2;
    task2->next = task4;
    task4->next = task5;

    int total = 0;
    Task *t = task_head;
    while (t != NULL) {
        if (t->id >= 0)
            ++total;
        t = t->next;
    }
    printf("We have a total number of %d tasks\n", total);

    // This corresponds to an empty task list.
    Task *empty_task_head = NULL;

    Task *task_evil = new Task(1, NULL);
    Task *task_2 = new Task(2, NULL);
    Task *task_3 = new Task(3, NULL);
    task_evil->next = task_2;
    task_2->next = task_3;
    task_3->next = task_evil; // In order to cause inifinite loop. :-)

    return 0; // Break at this line
}
