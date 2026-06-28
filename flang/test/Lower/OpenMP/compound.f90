! This test checks lowering of compound (combined and composite) constructs.
! Specifically, it makes sure that the proper ComposableOpInterface attributes
! are set.

! RUN: bbc -fopenmp -fopenmp-version=60 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! ------------------------------------------------------------------------------
! COMPOSITE CONSTRUCTS
! ------------------------------------------------------------------------------

subroutine distribute_parallel_do()
  implicit none
  integer :: i

  !$omp teams
  !$omp distribute parallel do
  do i=1, 10
  end do
  !$omp end teams
end subroutine

! CHECK-LABEL: func.func @_QPdistribute_parallel_do
! CHECK:         omp.parallel
! CHECK:           omp.distribute
! CHECK-NEXT:        omp.wsloop
! CHECK-NEXT:          omp.loop_nest
! CHECK:                 omp.yield
! CHECK-NEXT:          }
! CHECK-NEXT:        } {{{.*}}omp.composite{{.*}}}
! CHECK-NEXT:      } {{{.*}}omp.composite{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.composite{{.*}}}

subroutine distribute_parallel_do_simd()
  implicit none
  integer :: i

  !$omp teams
  !$omp distribute parallel do simd
  do i=1, 10
  end do
  !$omp end teams
end subroutine

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd
! CHECK:         omp.parallel
! CHECK:           omp.distribute
! CHECK-NEXT:        omp.wsloop
! CHECK-NEXT:          omp.simd
! CHECK-NEXT:            omp.loop_nest
! CHECK:                   omp.yield
! CHECK-NEXT:            }
! CHECK-NEXT:          } {{{.*}}omp.composite{{.*}}}
! CHECK-NEXT:        } {{{.*}}omp.composite{{.*}}}
! CHECK-NEXT:      } {{{.*}}omp.composite{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.composite{{.*}}}

subroutine distribute_simd()
  implicit none
  integer :: i

  !$omp teams
  !$omp distribute simd
  do i=1, 10
  end do
  !$omp end teams
end subroutine

! CHECK-LABEL: func.func @_QPdistribute_simd
! CHECK:         omp.distribute
! CHECK-NEXT:      omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      } {{{.*}}omp.composite{{.*}}}
! CHECK-NEXT:    } {{{.*}}omp.composite{{.*}}}

subroutine do_simd()
  implicit none
  integer :: i

  !$omp do simd
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPdo_simd
! CHECK:         omp.wsloop
! CHECK-NEXT:      omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      } {{{.*}}omp.composite{{.*}}}
! CHECK-NEXT:    } {{{.*}}omp.composite{{.*}}}

! TODO: Add taskloop simd once supported by lowering.

! ------------------------------------------------------------------------------
! COMBINED CONSTRUCTS
! ------------------------------------------------------------------------------

subroutine masked_taskloop()
  implicit none
  integer :: i

  !$omp masked taskloop
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPmasked_taskloop
! CHECK:         omp.masked
! CHECK:           omp.taskloop.context
! CHECK:             omp.taskloop.wrapper
! CHECK-NEXT:          omp.loop_nest
! CHECK:                 omp.yield
! CHECK-NEXT:          }
! CHECK-NEXT:        }
! CHECK-NOT:         omp.combined
! CHECK:             omp.terminator
! CHECK-NEXT:      } {{{.*}}omp.combined{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_taskloop()
  implicit none
  integer :: i

  !$omp master taskloop
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPmaster_taskloop
! CHECK:         omp.master
! CHECK:           omp.taskloop.context
! CHECK:             omp.taskloop.wrapper
! CHECK-NEXT:          omp.loop_nest
! CHECK:                 omp.yield
! CHECK-NEXT:          }
! CHECK-NEXT:        }
! CHECK-NOT:         omp.combined
! CHECK:             omp.terminator
! CHECK-NEXT:      } {{{.*}}omp.combined{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_do()
  implicit none
  integer :: i

  !$omp parallel do
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPparallel_do
! CHECK:         omp.parallel
! CHECK:           omp.wsloop
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_loop()
  implicit none
  integer :: i

  !$omp parallel loop
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPparallel_loop
! CHECK:         omp.parallel
! CHECK:           omp.wsloop
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_masked()
  implicit none

  !$omp parallel masked
  call foo()
  !$omp end parallel masked
end subroutine

! CHECK-LABEL: func.func @_QPparallel_masked
! CHECK:         omp.parallel
! CHECK:           omp.masked
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_master()
  implicit none

  !$omp parallel master
  call foo()
  !$omp end parallel master
end subroutine

! CHECK-LABEL: func.func @_QPparallel_master
! CHECK:         omp.parallel
! CHECK:           omp.master
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_sections()
  implicit none

  !$omp parallel sections
  call foo()
  !$omp end parallel sections
end subroutine

! CHECK-LABEL: func.func @_QPparallel_sections
! CHECK:         omp.parallel
! CHECK:           omp.sections
! CHECK:             omp.section
! CHECK:               omp.terminator
! CHECK-NEXT:        }
! CHECK:           }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_workshare()
  implicit none
  integer :: x(10)

  !$omp parallel workshare
  x = 1
  !$omp end parallel workshare
end subroutine

! CHECK-LABEL: func.func @_QPparallel_workshare
! CHECK:         omp.parallel
! CHECK:           omp.workshare
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_loop()
  implicit none
  integer :: i

  !$omp target loop
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_loop
! CHECK:         omp.target
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_parallel()
  implicit none

  !$omp target parallel
  call foo()
  !$omp end target parallel
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel
! CHECK:         omp.target
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_simd()
  implicit none
  integer :: i

  !$omp target simd
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_simd
! CHECK:         omp.target
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_teams()
  implicit none

  !$omp target teams
  call foo()
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams
! CHECK:         omp.target
! CHECK:           omp.teams
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine teams_distribute()
  implicit none
  integer :: i

  !$omp teams distribute
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPteams_distribute
! CHECK:         omp.teams
! CHECK:           omp.distribute
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine teams_loop()
  implicit none
  integer :: i

  !$omp teams loop
  do i=1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPteams_loop
! CHECK:         omp.teams
! CHECK:           omp.parallel
! CHECK:             omp.distribute
! CHECK-NEXT:          omp.wsloop
! CHECK-NEXT:            omp.loop_nest
! CHECK:                   omp.yield
! CHECK-NEXT:            }
! CHECK-NEXT:          } {{{.*}}omp.composite{{.*}}}
! CHECK-NEXT:        } {{{.*}}omp.composite{{.*}}}
! CHECK:             omp.terminator
! CHECK-NEXT:      } {{{.*}}omp.composite{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine teams_workdistribute()
  implicit none
  integer :: x

  !$omp teams workdistribute
  x = 1
  !$omp end teams workdistribute
end subroutine

! CHECK-LABEL: func.func @_QPteams_workdistribute
! CHECK:         omp.teams
! CHECK:           omp.workdistribute
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

! ------------------------------------------------------------------------------
! COMBINED CONSTRUCTS (SPLIT)
! ------------------------------------------------------------------------------

subroutine masked_loop()
  implicit none
  integer :: i

  !$omp masked
  !$omp loop
  do i=1, 10
  end do
  !$omp end masked
end subroutine

! CHECK-LABEL: func.func @_QPmasked_loop
! CHECK:         omp.masked
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine masked_parallel()
  implicit none

  !$omp masked
  !$omp parallel
  call foo() 
  !$omp end parallel
  !$omp end masked
end subroutine

! CHECK-LABEL: func.func @_QPmasked_parallel
! CHECK:         omp.masked
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine masked_simd()
  implicit none
  integer :: i

  !$omp masked
  !$omp simd
  do i=1, 10
  end do
  !$omp end masked
end subroutine

! CHECK-LABEL: func.func @_QPmasked_simd
! CHECK:         omp.masked
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine masked_target()
  implicit none

  !$omp masked
  !$omp target
  call foo()
  !$omp end target
  !$omp end masked
end subroutine

! CHECK-LABEL: func.func @_QPmasked_target
! CHECK:         omp.masked
! CHECK:           omp.target
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine masked_target_data()
  implicit none
  integer :: x(10)

  !$omp masked
  !$omp target_data map(tofrom: x)
  call foo()
  !$omp end target_data
  !$omp end masked
end subroutine

! CHECK-LABEL: func.func @_QPmasked_target_data
! CHECK:         omp.masked
! CHECK:           omp.target_data
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine masked_task()
  implicit none

  !$omp masked
  !$omp task
  call foo()
  !$omp end task
  !$omp end masked
end subroutine

! CHECK-LABEL: func.func @_QPmasked_task
! CHECK:         omp.masked
! CHECK:           omp.task
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_loop()
  implicit none
  integer :: i

  !$omp master
  !$omp loop
  do i=1, 10
  end do
  !$omp end master
end subroutine

! CHECK-LABEL: func.func @_QPmaster_loop
! CHECK:         omp.master
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_parallel()
  implicit none

  !$omp master
  !$omp parallel
  call foo()
  !$omp end parallel
  !$omp end master
end subroutine

! CHECK-LABEL: func.func @_QPmaster_parallel
! CHECK:         omp.master
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_simd()
  implicit none
  integer :: i

  !$omp master
  !$omp simd
  do i=1, 10
  end do
  !$omp end master
end subroutine

! CHECK-LABEL: func.func @_QPmaster_simd
! CHECK:         omp.master
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_target()
  implicit none

  !$omp master
  !$omp target
  call foo()
  !$omp end target
  !$omp end master
end subroutine

! CHECK-LABEL: func.func @_QPmaster_target
! CHECK:         omp.master
! CHECK:           omp.target
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_target_data()
  implicit none
  integer :: x(10)

  !$omp master
  !$omp target_data map(tofrom: x)
  call foo()
  !$omp end target_data
  !$omp end master
end subroutine

! CHECK-LABEL: func.func @_QPmaster_target_data
! CHECK:         omp.master
! CHECK:           omp.target_data
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine master_task()
  implicit none

  !$omp master
  !$omp task
  call foo()
  !$omp end task
  !$omp end master
end subroutine

! CHECK-LABEL: func.func @_QPmaster_task
! CHECK:         omp.master
! CHECK:           omp.task
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_simd()
  implicit none
  integer :: i

  !$omp parallel
  !$omp simd
  do i=1, 10
  end do
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparallel_simd
! CHECK:         omp.parallel
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_single()
  implicit none

  !$omp parallel
  !$omp single
  call foo()
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparallel_single
! CHECK:         omp.parallel
! CHECK:           omp.single
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_target()
  implicit none

  !$omp parallel
  !$omp target
  call foo()
  !$omp end target
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparallel_target
! CHECK:         omp.parallel
! CHECK:           omp.target
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_target_data()
  implicit none
  integer :: x(10)

  !$omp parallel
  !$omp target_data map(tofrom: x)
  call foo()
  !$omp end target_data
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparallel_target_data
! CHECK:         omp.parallel
! CHECK:           omp.target_data
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_task()
  implicit none

  !$omp parallel
  !$omp task
  call foo()
  !$omp end task
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparallel_task
! CHECK:         omp.parallel
! CHECK:           omp.task
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine parallel_taskloop()
  implicit none
  integer :: i

  !$omp parallel
  !$omp taskloop
  do i=1, 10
  end do
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparallel_taskloop
! CHECK:         omp.parallel
! CHECK:           omp.taskloop.context
! CHECK:             omp.taskloop.wrapper
! CHECK-NEXT:          omp.loop_nest
! CHECK:                 omp.yield
! CHECK-NEXT:          }
! CHECK-NEXT:        }
! CHECK-NOT:         omp.combined
! CHECK:             omp.terminator
! CHECK-NEXT:      } {{{.*}}omp.combined{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_loop()
  implicit none
  integer :: i

  !$omp single
  !$omp loop
  do i=1, 10
  end do
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_loop
! CHECK:         omp.single
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_parallel()
  implicit none

  !$omp single
  !$omp parallel
  call foo()
  !$omp end parallel
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_parallel
! CHECK:         omp.single
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_simd()
  implicit none
  integer :: i

  !$omp single
  !$omp simd
  do i=1, 10
  end do
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_simd
! CHECK:         omp.single
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_target()
  implicit none

  !$omp single
  !$omp target
  call foo()
  !$omp end target
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_target
! CHECK:         omp.single
! CHECK:           omp.target
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_target_data()
  implicit none
  integer :: x(10)

  !$omp single
  !$omp target_data map(tofrom: x)
  call foo()
  !$omp end target_data
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_target_data
! CHECK:         omp.single
! CHECK:           omp.target_data
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_task()
  implicit none

  !$omp single
  !$omp task
  call foo()
  !$omp end task
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_task
! CHECK:         omp.single
! CHECK:           omp.task
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine single_taskloop()
  implicit none
  integer :: i

  !$omp single
  !$omp taskloop
  do i=1, 10
  end do
  !$omp end single
end subroutine

! CHECK-LABEL: func.func @_QPsingle_taskloop
! CHECK:         omp.single
! CHECK:           omp.taskloop.context
! CHECK:             omp.taskloop.wrapper
! CHECK-NEXT:          omp.loop_nest
! CHECK:                 omp.yield
! CHECK-NEXT:          }
! CHECK-NEXT:        }
! CHECK-NOT:         omp.combined
! CHECK:             omp.terminator
! CHECK-NEXT:      } {{{.*}}omp.combined{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_task()
  implicit none

  !$omp target
  !$omp task
  call foo()
  !$omp end task
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_task
! CHECK:         omp.target
! CHECK:           omp.task
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_taskloop()
  implicit none
  integer :: i

  !$omp target
  !$omp taskloop
  do i=1, 10
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_taskloop
! CHECK:         omp.target
! CHECK:           omp.taskloop.context
! CHECK:             omp.taskloop.wrapper
! CHECK-NEXT:          omp.loop_nest
! CHECK:                 omp.yield
! CHECK-NEXT:          }
! CHECK-NEXT:        }
! CHECK-NOT:         omp.combined
! CHECK:             omp.terminator
! CHECK-NEXT:      } {{{.*}}omp.combined{{.*}}}
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_data_loop()
  implicit none
  integer :: i
  integer :: x(10)

  !$omp target_data map(tofrom: x)
  !$omp loop
  do i=1, 10
  end do
  !$omp end target_data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_loop
! CHECK:         omp.target_data
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_data_parallel()
  implicit none
  integer :: x(10)

  !$omp target_data map(tofrom: x)
  !$omp parallel
  call foo()
  !$omp end parallel
  !$omp end target_data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_parallel
! CHECK:         omp.target_data
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine target_data_simd()
  implicit none
  integer :: i
  integer :: x(10)

  !$omp target_data map(tofrom: x)
  !$omp simd
  do i=1, 10
  end do
  !$omp end target_data
end subroutine

! CHECK-LABEL: func.func @_QPtarget_data_simd
! CHECK:         omp.target_data
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine task_loop()
  implicit none
  integer :: i

  !$omp task
  !$omp loop
  do i=1, 10
  end do
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_loop
! CHECK:         omp.task
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine task_parallel()
  implicit none

  !$omp task
  !$omp parallel
  call foo()
  !$omp end parallel
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_parallel
! CHECK:         omp.task
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine task_simd()
  implicit none
  integer :: i

  !$omp task
  !$omp simd
  do i=1, 10
  end do
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPtask_simd
! CHECK:         omp.task
! CHECK:           omp.simd
! CHECK-NEXT:        omp.loop_nest
! CHECK:               omp.yield
! CHECK-NEXT:        }
! CHECK-NEXT:      }
! CHECK-NOT:       omp.combined
! CHECK:           omp.terminator
! CHECK-NEXT:    } {{{.*}}omp.combined{{.*}}}

subroutine teams_parallel()
  implicit none

  !$omp teams
  !$omp parallel
  call foo()
  !$omp end parallel
  !$omp end teams
end subroutine
