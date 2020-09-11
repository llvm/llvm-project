# OpenMP 4.5 Grammar

Grammar used by Flang to parse OpenMP 4.5.

## OpenMP 4.5 Specifications
```
2 omp-directive -> sentinel directive-name [clause[ [,] clause]...]
2.1.1 sentinel -> !$omp | c$omp | *$omp
2.1.2 sentinel -> !$omp
```

## directive-name
```
2.5 parallel -> PARALLEL [parallel-clause[ [,] parallel-clause]...]
    parallel-clause -> if-clause |
                       num-threads-clause |
                       default-clause |
                       private-clause |
                       firstprivate-clause |
                       shared-clause |
                       copyin-clause |
                       reduction-clause |
                       proc-bind-clause

2.5 end-parallel -> END PARALLEL

2.7.1 do -> DO [do-clause[ [,] do-clause]...]
      do-clause -> private-clause |
                   firstprivate-clause |
                   lastprivate-clause |
                   linear-clause |
                   reduction-clause |
                   schedule-clause |
                   collapse-clause |
                   ordered-clause

2.7.1 end-do -> END DO [nowait-clause]

2.7.2 sections -> SECTIONS [sections-clause[ [,] sections-clause]...]
      sections-clause -> private-clause |
                         firstprivate-clause |
                         lastprivate-clause |
                         reduction-clause

2.7.2 section -> SECTION

2.7.2 end-sections -> END SECTIONS [nowait-clause]

2.7.3 single -> SINGLE [single-clause[ [,] single-clause]...]
      single-clause -> private-clause |
                       firstprivate-clause

2.7.3 end-single -> END SINGLE [end-single-clause[ [,] end-single-clause]...]
      end-single-clause -> copyprivate-clause |
                           nowait-clause

2.7.4 workshare -> WORKSHARE

2.7.4 end-workshare -> END WORKSHARE [nowait-clause]

2.8.1 simd -> SIMD [simd-clause[ [,] simd-clause]...]
      simd-clause -> safelen-clause |
                     simdlen-clause |
                     linear-clause |
                     aligned-clause |
                     private-clause |
                     lastprivate-clause |
                     reduction-clause |
                     collapse-clause

2.8.1 end-simd -> END SIMD

2.8.2 declare-simd -> DECLARE SIMD [(proc-name)] [declare-simd-clause[ [,] declare-simd-clause]...]
      declare-simd-clause -> simdlen-clause |
                             linear-clause |
                             aligned-clause |
                             uniform-clause |
                             inbranch-clause |
                             notinbranch-clause

2.8.3 do-simd -> DO SIMD [do-simd-clause[ [,] do-simd-clause]...]
      do-simd-clause -> do-clause |
                        simd-clause

2.8.3 end-do-simd -> END DO SIMD [nowait-clause]

2.9.1 task -> TASK [task-clause[ [,] task-clause]...]
      task-clause -> if-clause |
                     final-clause |
                     untied-clause |
                     default-clause |
                     mergeable-clause |
                     private-clause |
                     firstprivate-clause |
                     shared-clause |
                     depend-clause |
                     priority-clause

2.9.1 end-task -> END TASK

2.9.2 taskloop -> TASKLOOP [taskloop-clause[ [,] taskloop-clause]...]
      taskloop-clause -> if-clause |
                         shared-clause |
                         private-clause |
                         firstprivate-clause |
                         lastprivate-clause |
                         default-clause |
                         grainsize-clause |
                         num-tasks-clause |
                         collapse-clause |
                         final-clause |
                         priority-clause |
                         untied-clause |
                         mergeable-clause |
                         nogroup-clause

2.9.2 end-taskloop -> END TASKLOOP

2.9.3 taskloop-simd -> TASKLOOP SIMD [taskloop-simd-clause[ [,] taskloop-simd-clause]...]
      taskloop-simd-clause -> taskloop-clause |
                              simd-clause

2.9.3 end-taskloop-simd -> END TASKLOOP SIMD

2.9.4 taskyield -> TASKYIELD

2.10.1 target-data -> TARGET DATA target-data-clause[ [ [,] target-data-clause]...]
       target-data-clause -> if-clause |
                             device-clause |
                             map-clause |
                             use-device-ptr-clause

2.10.1 end-target-data -> END TARGET DATA

2.10.2 target-enter-data -> TARGET ENTER DATA [ target-enter-data-clause[ [,] target-enter-data-clause]...]
       target-enter-data-clause -> if-clause |
                                   device-clause |
                                   map-clause |
                                   depend-clause |
                                   nowait-clause

2.10.3 target-exit-data -> TARGET EXIT DATA [ target-exit-data-clause[ [,] target-exit-data-clause]...]
       target-exit-data-clause -> if-clause |
                                  device-clause |
                                  map-clause |
                                  depend-clause |
                                  nowait-clause

2.10.4 target -> TARGET [target-clause[ [,] target-clause]...]
       target-clause -> if-clause |
                        device-clause |
                        private-clause |
                        firstprivate-clause |
                        map-clause |
                        is-device-ptr-clause |
                        defaultmap-clause |
                        nowait-clause |
                        depend-clause

2.10.4 end-target -> END TARGET

2.10.5 target-update -> TARGET UPDATE target-update-clause[ [ [,] target-update-clause]...]
       target-update-clause -> motion-clause |
                               if-clause |
                               device-clause |
                               nowait-clause |
                               depend-clause
       motion-clause -> to-clause |
                        from-clause

2.10.6 declare-target -> DECLARE TARGET (extended-list) |
                         DECLARE TARGET [declare-target-clause[ [,] declare-target-clause]...]
       declare-target-clause -> to-clause |
                                link-clause

2.10.7 teams -> TEAMS [teams-clause[ [,] teams-clause]...]
       teams-clause -> num-teams-clause |
                       thread-limit-clause |
                       default-clause |
                       private-clause |
                       firstprivate-clause |
                       shared-clause |
                       reduction-clause

2.10.7 end-teams -> END TEAMS

2.10.8 distribute -> DISTRIBUTE [distribute-clause[ [,] distribute-clause]...]
       distribute-clause -> private-clause |
                            firstprivate-clause |
                            lastprivate-clause |
                            collapse-clause |
                            dist-schedule-clause

2.10.8 end-distribute -> END DISTRIBUTE

2.10.9 distribute-simd -> DISTRIBUTE SIMD [distribute-simd-clause[ [,] distribute-simd-clause]...]
       distribute-simd-clause -> distribute-clause |
                                 simd-clause

2.10.9 end-distribute-simd -> END DISTRIBUTE SIMD

2.10.10 distribute-parellel-do ->
           DISTRIBUTE PARALLEL DO [distribute-parallel-do-clause[ [,] distribute-parallel-do-clause]...]
        distribute-parallel-do-clause -> distribute-clause |
                                         parallel-do-clause

2.10.10 end-distribute-parellel-do -> END DISTRIBUTE PARALLEL DO

2.10.11 distribute-parallel-do-simd ->
           DISTRIBUTE PARALLEL DO SIMD [distribute-parallel-do-simd-clause[ [,] distribute-parallel-do-simd-clause]...]
        distribute-parallel-do-simd-clause -> distribute-clause |
                                              parallel-do-simd-clause

2.10.11 end-distribute-parallel-do-simd -> END DISTRIBUTE PARALLEL DO SIMD

2.11.1 parallel-do -> PARALLEL DO [parallel-do-clause[ [,] parallel-do-clause]...]
       parallel-do-clause -> parallel-clause |
                             do-clause

2.11.1 end-parallel-do -> END PARALLEL DO

2.11.2 parallel-sections -> PARALLEL SECTIONS [parallel-sections-clause[ [,] parallel-sections-clause]...]
       parallel-sections-clause -> parallel-clause |
                                   sections-clause

2.11.2 end-parallel-sections -> END PARALLEL SECTIONS

2.11.3 parallel-workshare -> PARALLEL WORKSHARE [parallel-workshare-clause[ [,] parallel-workshare-clause]...]
       parallel-workshare-clause -> parallel-clause

2.11.3 end-parallel-workshare -> END PARALLEL WORKSHARE

2.11.4 parallel-do-simd -> PARALLEL DO SIMD [parallel-do-simd-clause[ [,] parallel-do-simd-clause]...]
       parallel-do-simd-clause -> parallel-clause |
                                  do-simd-clause

2.11.4 end-parallel-do-simd -> END PARALLEL DO SIMD

2.11.5 target-parallel -> TARGET PARALLEL [target-parallel-clause[ [,] target-parallel-clause]...]
       target-parallel-clause -> target-clause |
                                 parallel-clause

2.11.5 end-target-parallel -> END TARGET PARALLEL

2.11.6 target-parallel-do -> TARGET PARALLEL DO [target-parallel-do-clause[ [,] target-parallel-do-clause]...]
       target-parallel-do-clause -> target-clause |
                                    parallel-do-clause

2.11.6 end-target-parallel-do -> END TARGET PARALLEL DO

2.11.7 target-parallel-do-simd ->
          TARGET PARALLEL DO SIMD [target-parallel-do-simd-clause[ [,] target-parallel-do-simd-clause]...]
       target-parallel-do-simd-clause -> target-clause |
                                         parallel-do-simd-clause

2.11.7 end-target-parallel-do-simd -> END TARGET PARALLEL DO SIMD

2.11.8 target-simd -> TARGET SIMD [target-simd-clause[ [,] target-simd-clause]...]
       target-simd-clause -> target-clause |
                             simd-clause

2.11.8 end-target-simd -> END TARGET SIMD

2.11.9 target-teams -> TARGET TEAMS [target-teams-clause[ [,] target-teams-clause]...]
       target-teams-clause -> target-clause |
                              teams-clause

2.11.9 end-target-teams -> END TARGET TEAMS

2.11.10 teams-distribute -> TEAMS DISTRIBUTE [teams-distribute-clause[ [,] teams-distribute-clause]...]
        teams-distribute-clause -> teams-clause |
                                   distribute-clause

2.11.10 end-teams-distribute -> END TEAMS DISTRIBUTE

2.11.11 teams-distribute-simd ->
           TEAMS DISTRIBUTE SIMD [teams-distribute-simd-clause[ [,] teams-distribute-simd-clause]...]
        teams-distribute-simd-clause -> teams-clause |
                                        distribute-simd-clause

2.11.11 end-teams-distribute-simd -> END TEAMS DISTRIBUTE SIMD

2.11.12 target-teams-distribute ->
           TARGET TEAMS DISTRIBUTE [target-teams-distribute-clause[ [,] target-teams-distribute-clause]...]
        target-teams-distribute-clause -> target-clause |
                                          teams-distribute-clause

2.11.12 end-target-teams-distribute -> END TARGET TEAMS DISTRIBUTE

2.11.13 target-teams-distribute-simd ->
           TARGET TEAMS DISTRIBUTE SIMD [target-teams-distribute-simd-clause[ [,] target-teams-distribute-simd-clause]...]
        target-teams-distribute-simd-clause -> target-clause |
                                               teams-distribute-simd-clause

2.11.13 end-target-teams-distribute-simd -> END TARGET TEAMS DISTRIBUTE SIMD

2.11.14 teams-distribute-parallel-do ->
           TEAMS DISTRIBUTE PARALLEL DO [teams-distribute-parallel-do-clause[ [,] teams-distribute-parallel-do-clause]...]
        teams-distribute-parallel-do-clause -> teams-clause |
                                               distribute-parallel-do-clause

2.11.14 end-teams-distribute-parallel-do -> END TEAMS DISTRIBUTE PARALLEL DO

2.11.15 target-teams-distribute-parallel-do ->
           TARGET TEAMS DISTRIBUTE PARALLEL DO [target-teams-distribute-parallel-do-clause[ [,] target-teams-distribute-parallel-do-clause]...]
        target-teams-distribute-parallel-do-clause -> target-clause |
                                                      teams-distribute-parallel-do-clause

2.11.15 end-target-teams-distribute-parallel-do -> END TARGET TEAMS DISTRIBUTE PARALLEL DO

2.11.16 teams-distribute-parallel-do-simd ->
           TEAMS DISTRIBUTE PARALLEL DO SIMD [teams-distribute-parallel-do-simd-clause[ [,] teams-distribute-parallel-do-simd-clause]...]
        teams-distribute-parallel-do-simd-clause -> teams-clause |
                                                    distribute-parallel-do-simd-clause

2.11.16 end-teams-distribute-parallel-do-simd -> END TEAMS DISTRIBUTE PARALLEL DO SIMD

2.11.17 target-teams-distribute-parallel-do-simd ->
           TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD [target-teams-distribute-parallel-do-simd-clause[ [,] target-teams-distribute-parallel-do-simd-clause]...]
        target-teams-distribute-parallel-do-simd-clause -> target-clause |
                                                           teams-distribute-parallel-do-simd-clause

2.11.17 end-target-teams-distribute-parallel-do-simd -> END TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD

2.13.1 master -> MASTER

2.13.1 end-master -> END MASTER

2.13.2 critical -> CRITICAL [(name) [HINT(hint-expr)]]

2.13.2 end-critical -> END CRITICAL [(name)]

2.13.3 barrier -> BARRIER

2.13.4 taskwait -> TASKWAIT

2.13.5 taskgroup -> TASKGROUP

2.13.5 end-taskgroup -> END TASKGROUP

2.13.6 atomic -> ATOMIC [seq_cst[,]] atomic-clause [[,]seq_cst] |
                 ATOMIC [seq_cst]
       atomic-clause -> READ | WRITE | UPDATE | CAPTURE

2.13.7 flush -> FLUSH [(variable-name-list)]

2.13.8 ordered -> ORDERED ordered-construct-clause [[[,] ordered-construct-clause]...]
       ordered-construct-clause -> depend-clause

2.13.8 end-ordered -> END ORDERED

2.14.1 cancel -> CANCEL construct-type-clause [ [,] if-clause]
       construct-type-clause -> PARALLEL |
                                SECTIONS |
                                DO |
                                TASKGROUP

2.14.2 cancellation-point -> CANCELLATION POINT construct-type-clause

2.15.2 threadprivate -> THREADPRIVATE (variable-name-list)

2.16 declare-reduction -> DECLARE REDUCTION (reduction-identifier : type-list : combiner) [initializer-clause]

# Clauses
2.5 proc-bind-clause -> PROC_BIND (MASTER | CLOSE | SPREAD)

2.5 num-threads-clause -> NUM_THREADS (scalar-int-expr)

2.7.1 schedule-clause -> SCHEDULE ([sched-modifier] [, sched-modifier]:]
                                   kind[, chunk_size])

2.7.1 kind -> STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME

2.7.1 sched-modifier -> MONOTONIC | NONMONOTONIC | SIMD

2.7.1 chunk_size -> scalar-int-expr

2.7.1 collapse-clause -> COLLAPSE (scalar-constant)

2.7.1 ordered-clause -> ORDERED [(scalar-constant)]

2.7.1 nowait-clause -> NOWAIT

2.8.1 aligned-clause -> ALIGNED (variable-name-list[ : scalar-constant])

2.8.1 safelen-clause -> SAFELEN (scalar-constant)

2.8.1 simdlen-clause -> SIMDLEN (scalar-contant)

2.8.2 uniform-clause -> UNIFORM (dummy-arg-name-list)

2.8.2 inbranch-clause -> INBRANCH

2.8.2 notinbranch-clause -> NOTINBRANCH

2.13.9 depend-clause -> DEPEND (((IN | OUT | INOUT) : variable-name-list) |
                                SOURCE |
                                SINK : vec)
                 vec -> iterator [+/- scalar-int-expr],..., iterator[...]

2.9.2 num-tasks-clause -> NUM_TASKS (scalar-int-expr)

2.9.2 grainsize-clause -> GRAINSIZE (scalar-int-expr)

2.9.2 nogroup-clause -> NOGROUP

2.9.2 untied-clause -> UNTIED

2.9.2 priority-clause -> PRIORITY (scalar-int-expr)

2.9.2 mergeable-clause -> MERGEABLE

2.9.2 final-clause -> FINAL (scalar-int-expr)

2.10.1 use-device-ptr-clause -> USE_DEVICE_PTR (variable-name-list)

2.10.1 device-clause -> DEVICE (scalar-integer-expr)

2.10.4 is-device-ptr-clause -> IS_DEVICE_PTR (variable-name-list)

2.10.5 to-clause -> TO (variable-name-list)

2.10.5 from-clause -> FROM (variable-name-list)

2.10.6 link-clause -> LINK (variable-name-list)

2.10.7 num-teams-clause -> NUM_TEAMS (scalar-integer-expr)

2.10.7 thread-limit-clause -> THREAD_LIMIT (scalar-integer-expr)

2.10.8 dist-schedule-clause -> DIST_SCHEDULE (STATIC [ , chunk_size])

2.12 if-clause -> IF ([ directive-name-modifier :] scalar-logical-expr)

2.15.3.1 default-clause -> DEFAULT (PRIVATE | FIRSTPRIVATE | SHARED | NONE)

2.15.3.2 shared-clause -> SHARED (variable-name-list)

2.15.3.3 private-clause -> PRIVATE (variable-name-list)

2.15.3.4 firstprivate-clause -> FIRSTPRIVATE (variable-name-list)

2.15.3.5 lastprivate-clause -> LASTPRIVATE (variable-name-list)

2.15.3.6 reduction-clause -> REDUCTION (reduction-identifier: variable-name-list)
         reduction-identifier -> + | - | * |
                                 .AND. | .OR. | .EQV. | .NEQV. |
                                 MAX | MIN | IAND | IOR | IEOR

2.15.3.7 linear-clause -> LINEAR (linear-list[ : linear-step])
         linear-list -> list | modifier(list)
         modifier -> REF | VAL | UVAL

2.15.4.1 copyin-clause -> COPYIN (variable-name-list)

2.15.4.2 copyprivate-clause -> COPYPRIVATE (variable-name-list)

2.15.5.1 map -> MAP ([ [ALWAYS[,]] map-type : ] variable-name-list)
         map-type -> TO | FROM | TOFROM |
                     ALLOC | RELEASE | DELETE

2.15.5.2 defaultmap -> DEFAULTMAP (TOFROM:SCALAR)
```
