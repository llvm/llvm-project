! RUN: bbc -emit-fir -o - %s | FileCheck %s

  ! CHECK-LABEL: sinteger
  function sinteger(n)
    integer sinteger
    nn = -88
    ! CHECK: fir.select_case {{.*}} : i32
    ! CHECK-SAME: upper, %c1
    ! CHECK-SAME: point, %c2
    ! CHECK-SAME: point, %c3
    ! CHECK-SAME: interval, %c4{{.*}} %c5
    ! CHECK-SAME: point, %c6
    ! CHECK-SAME: point, %c7
    ! CHECK-SAME: interval, %c8{{.*}} %c15
    ! CHECK-SAME: lower, %c21
    ! CHECK-SAME: unit
    select case(n)
    case (:1)
      nn = 1
    case (2)
      nn = 2
    case default
      nn = 0
    case (3)
      nn = 3
    case (4:5+1-1)
      nn = 4
    case (6)
      nn = 6
    case (7,8:15,21:)
      nn = 7
    end select
    sinteger = nn
  end

  ! CHECK-LABEL: slogical
  subroutine slogical(L)
    logical :: L
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0
    n8 = 0

    select case (L)
    end select

    select case (L)
      ! CHECK: cmpi eq, {{.*}} %false
      ! CHECK: cond_br
      case (.false.)
        n2 = 1
    end select

    select case (L)
      ! CHECK: cmpi eq, {{.*}} %true
      ! CHECK: cond_br
      case (.true.)
        n3 = 2
    end select

    select case (L)
      case default
        n4 = 3
    end select

    select case (L)
      ! CHECK: cmpi eq, {{.*}} %false
      ! CHECK: cond_br
      case (.false.)
        n5 = 1
      ! CHECK: cmpi eq, {{.*}} %true
      ! CHECK: cond_br
      case (.true.)
        n5 = 2
    end select

    select case (L)
      ! CHECK: cmpi eq, {{.*}} %false
      ! CHECK: cond_br
      case (.false.)
        n6 = 1
      case default
        n6 = 3
    end select

    select case (L)
      ! CHECK: cmpi eq, {{.*}} %true
      ! CHECK: cond_br
      case (.true.)
        n7 = 2
      case default
        n7 = 3
    end select

    select case (L)
      ! CHECK: cmpi eq, {{.*}} %false
      ! CHECK: cond_br
      case (.false.)
        n8 = 1
      ! CHECK: cmpi eq, {{.*}} %true
      ! CHECK: cond_br
      case (.true.)
        n8 = 2
      ! CHECK-NOT: constant 888
      case default ! dead
        n8 = 888
    end select

    print*, n1, n2, n3, n4, n5, n6, n7, n8
  end

  ! CHECK-LABEL: scharacter
  subroutine scharacter(c)
    character(*) :: c
    nn = 0
    select case (c)
      case default
        nn = -1
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: constant 0
      ! CHECK-NEXT: cmpi sle, {{.*}} %c0
      ! CHECK-NEXT: cond_br
      case (:'d')
        nn = 10
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: constant 0
      ! CHECK-NEXT: cmpi sge, {{.*}} %c0
      ! CHECK-NEXT: cond_br
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: constant 0
      ! CHECK-NEXT: cmpi sle, {{.*}} %c0
      ! CHECK-NEXT: cond_br
      case ('ff':'ffff')
        nn = 20
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: constant 0
      ! CHECK-NEXT: cmpi eq, {{.*}} %c0
      ! CHECK-NEXT: cond_br
      case ('m')
        nn = 30
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: constant 0
      ! CHECK-NEXT: cmpi eq, {{.*}} %c0
      ! CHECK-NEXT: cond_br
      case ('qq')
        nn = 40
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: constant 0
      ! CHECK-NEXT: cmpi sge, {{.*}} %c0
      ! CHECK-NEXT: cond_br
      case ('x':)
        nn = 50
    end select
    print*, nn
  end

  ! CHECK-LABEL: func @_QPtest_char_temp_selector
  subroutine test_char_temp_selector()
    ! Test that character selector that are temps are deallocated
    ! only after they have been used in the select case comparisons.
    interface
      function gen_char_temp_selector()
        character(:), allocatable :: gen_char_temp_selector
      end function
    end interface
    select case (gen_char_temp_selector())
    case ('case1')
      call foo1()
    case ('case2')
      call foo2()
    case ('case3')
      call foo3()
    case default
      call foo_default()
    end select
    ! CHECK:   %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".result"}
    ! CHECK:   %[[VAL_1:.*]] = fir.call @_QPgen_char_temp_selector() : () -> !fir.box<!fir.heap<!fir.char<1,?>>>
    ! CHECK:   fir.save_result %[[VAL_1]] to %[[VAL_0]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    ! CHECK:   cond_br %{{.*}}, ^bb2, ^bb1
    ! CHECK: ^bb1:
    ! CHECK:   cond_br %{{.*}}, ^bb4, ^bb3
    ! CHECK: ^bb2:
    ! CHECK:   fir.call @_QPfoo1() : () -> ()
    ! CHECK:   br ^bb8
    ! CHECK: ^bb3:
    ! CHECK:   cond_br %{{.*}}, ^bb6, ^bb5
    ! CHECK: ^bb4:
    ! CHECK:   fir.call @_QPfoo2() : () -> ()
    ! CHECK:   br ^bb8
    ! CHECK: ^bb5:
    ! CHECK:   br ^bb7
    ! CHECK: ^bb6:
    ! CHECK:   fir.call @_QPfoo3() : () -> ()
    ! CHECK:   br ^bb8
    ! CHECK: ^bb7:
    ! CHECK:   fir.call @_QPfoo_default() : () -> ()
    ! CHECK:   br ^bb8
    ! CHECK: ^bb8:
    ! CHECK:   %[[VAL_36:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    ! CHECK:   %[[VAL_37:.*]] = fir.box_addr %[[VAL_36]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
    ! CHECK:   %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.heap<!fir.char<1,?>>) -> i64
    ! CHECK:   %[[VAL_39:.*]] = arith.constant 0 : i64
    ! CHECK:   %[[VAL_40:.*]] = arith.cmpi ne, %[[VAL_38]], %[[VAL_39]] : i64
    ! CHECK:   fir.if %[[VAL_40]] {
    ! CHECK:     fir.freemem %[[VAL_37]]
    ! CHECK:   }
  end subroutine
