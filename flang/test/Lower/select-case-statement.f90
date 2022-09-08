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

  ! CHECK-LABEL: func @_QPscharacter1
  subroutine scharacter1(s)
    ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
    character(len=3) :: s
    ! CHECK-DAG: %[[V_1:[0-9]+]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFscharacter1En"}
    ! CHECK:     fir.store %c0{{.*}} to %[[V_1]] : !fir.ref<i32>
    n = 0

    ! CHECK:     %[[V_8:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
    ! CHECK:     %[[V_9:[0-9]+]] = arith.cmpi sge, %[[V_8]], %c0{{.*}} : i32
    ! CHECK:     cond_br %[[V_9]], ^bb1, ^bb15
    ! CHECK:   ^bb1:  // pred: ^bb0
    if (lge(s,'00')) then

      ! CHECK:   %[[V_18:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
      ! CHECK:   %[[V_20:[0-9]+]] = fir.box_addr %[[V_18]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
      ! CHECK:   %[[V_42:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_43:[0-9]+]] = arith.cmpi eq, %[[V_42]], %c0{{.*}} : i32
      ! CHECK:   fir.if %[[V_43]] {
      ! CHECK:     fir.freemem %[[V_20]] : !fir.heap<!fir.char<1,?>>
      ! CHECK:   }
      ! CHECK:   cond_br %[[V_43]], ^bb3, ^bb2
      ! CHECK: ^bb2:  // pred: ^bb1
      select case(trim(s))
      case('11')
        n = 1

      case default
        continue

      ! CHECK:   %[[V_48:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_49:[0-9]+]] = arith.cmpi eq, %[[V_48]], %c0{{.*}} : i32
      ! CHECK:   fir.if %[[V_49]] {
      ! CHECK:     fir.freemem %[[V_20]] : !fir.heap<!fir.char<1,?>>
      ! CHECK:   }
      ! CHECK:   cond_br %[[V_49]], ^bb6, ^bb5
      ! CHECK: ^bb3:  // pred: ^bb1
      ! CHECK:   fir.store %c1{{.*}} to %[[V_1]] : !fir.ref<i32>
      ! CHECK: ^bb4:  // pred: ^bb13
      ! CHECK: ^bb5:  // pred: ^bb2
      case('22')
        n = 2

      ! CHECK:   %[[V_54:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_55:[0-9]+]] = arith.cmpi eq, %[[V_54]], %c0{{.*}} : i32
      ! CHECK:   fir.if %[[V_55]] {
      ! CHECK:     fir.freemem %[[V_20]] : !fir.heap<!fir.char<1,?>>
      ! CHECK:   }
      ! CHECK:   cond_br %[[V_55]], ^bb8, ^bb7
      ! CHECK: ^bb6:  // pred: ^bb2
      ! CHECK:   fir.store %c2{{.*}} to %[[V_1]] : !fir.ref<i32>
      ! CHECK: ^bb7:  // pred: ^bb5
      case('33')
        n = 3

      case('44':'55','66':'77','88':)
        n = 4
      ! CHECK:   %[[V_60:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_61:[0-9]+]] = arith.cmpi sge, %[[V_60]], %c0{{.*}} : i32
      ! CHECK:   cond_br %[[V_61]], ^bb9, ^bb10
      ! CHECK: ^bb8:  // pred: ^bb5
      ! CHECK:   fir.store %c3{{.*}} to %[[V_1]] : !fir.ref<i32>
      ! CHECK: ^bb9:  // pred: ^bb7
      ! CHECK:   %[[V_66:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_67:[0-9]+]] = arith.cmpi sle, %[[V_66]], %c0{{.*}} : i32
      ! CHECK:   fir.if %[[V_67]] {
      ! CHECK:     fir.freemem %[[V_20]] : !fir.heap<!fir.char<1,?>>
      ! CHECK:   }
      ! CHECK:   cond_br %[[V_67]], ^bb14, ^bb10
      ! CHECK: ^bb10:  // 2 preds: ^bb7, ^bb9
      ! CHECK:   %[[V_72:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_73:[0-9]+]] = arith.cmpi sge, %[[V_72]], %c0{{.*}} : i32
      ! CHECK:   cond_br %[[V_73]], ^bb11, ^bb12
      ! CHECK: ^bb11:  // pred: ^bb10
      ! CHECK:   %[[V_78:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_79:[0-9]+]] = arith.cmpi sle, %[[V_78]], %c0{{.*}} : i32
      ! CHECK:   fir.if %[[V_79]] {
      ! CHECK:     fir.freemem %[[V_20]] : !fir.heap<!fir.char<1,?>>
      ! CHECK:   }
      ! CHECK: ^bb12:  // 2 preds: ^bb10, ^bb11
      ! CHECK:   %[[V_84:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
      ! CHECK:   %[[V_85:[0-9]+]] = arith.cmpi sge, %[[V_84]], %c0{{.*}} : i32
      ! CHECK:   fir.freemem %[[V_20]] : !fir.heap<!fir.char<1,?>>
      ! CHECK:   cond_br %[[V_85]], ^bb14, ^bb13
      ! CHECK: ^bb13:  // pred: ^bb12
      ! CHECK: ^bb14:  // 3 preds: ^bb9, ^bb11, ^bb12
      ! CHECK:   fir.store %c4{{.*}} to %[[V_1]] : !fir.ref<i32>
      ! CHECK: ^bb15:  // 6 preds: ^bb0, ^bb3, ^bb4, ^bb6, ^bb8, ^bb14
      end select
    end if
    ! CHECK:     %[[V_89:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
    print*, n
  end subroutine


  ! CHECK-LABEL: func @_QPscharacter2
  subroutine scharacter2(s)
    ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
    ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
    character(len=3) :: s
    n = 0

    ! CHECK:   %[[V_12:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    ! CHECK:   %[[V_13:[0-9]+]] = fir.box_addr %[[V_12]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
    ! CHECK:   fir.freemem %[[V_13]] : !fir.heap<!fir.char<1,?>>
    ! CHECK:   br ^bb1
    ! CHECK: ^bb1:  // pred: ^bb0
    ! CHECK:   br ^bb2
    n = -10
    select case(trim(s))
    case default
      n = 9
    end select
    print*, n

    ! CHECK: ^bb2:  // pred: ^bb1
    ! CHECK:   %[[V_28:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
    ! CHECK:   %[[V_29:[0-9]+]] = fir.box_addr %[[V_28]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
    ! CHECK:   fir.freemem %[[V_29]] : !fir.heap<!fir.char<1,?>>
    ! CHECK:   br ^bb3
    ! CHECK: ^bb3:  // pred: ^bb2
    n = -2
    select case(trim(s))
    end select
    print*, n
  end subroutine

  ! CHECK-LABEL: func @_QPsempty
  ! empty select case blocks
  subroutine sempty(n)
    ! CHECK:   %[[selectI1:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK:   fir.select_case %[[selectI1]] : i32 [#fir.point, %c1{{.*}}, ^bb1, #fir.point, %c2{{.*}}, ^bb2, unit, ^bb3]
    ! CHECK: ^bb1:  // pred: ^bb0
    ! CHECK:   fir.call @_FortranAioBeginExternalListOutput
    ! CHECK:   br ^bb4
    ! CHECK: ^bb2:  // pred: ^bb0
    ! CHECK:   br ^bb4
    ! CHECK: ^bb3:  // pred: ^bb0
    ! CHECK:   fir.call @_FortranAioBeginExternalListOutput
    ! CHECK:   br ^bb4
    select case (n)
      case (1)
        print*, n, 'i:case 1'
      case (2)
      ! print*, n, 'i:case 2'
      case default
        print*, n, 'i:case default'
    end select
    ! CHECK: ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
    ! CHECK:   %[[cmpC1:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
    ! CHECK:   %[[selectC1:[0-9]+]] = arith.cmpi eq, %[[cmpC1]], %c0{{.*}} : i32
    ! CHECK:   cond_br %[[selectC1]], ^bb6, ^bb5
    ! CHECK: ^bb5:  // pred: ^bb4
    ! CHECK:   %[[cmpC2:[0-9]+]] = fir.call @_FortranACharacterCompareScalar1
    ! CHECK:   %[[selectC2:[0-9]+]] = arith.cmpi eq, %[[cmpC2]], %c0{{.*}} : i32
    ! CHECK:   cond_br %[[selectC2]], ^bb8, ^bb7
    ! CHECK: ^bb6:  // pred: ^bb4
    ! CHECK:   fir.call @_FortranAioBeginExternalListOutput
    ! print*, n, 'c:case 2'
    ! CHECK:   br ^bb10
    ! CHECK: ^bb7:  // pred: ^bb5
    ! CHECK:   br ^bb9
    ! CHECK: ^bb8:  // pred: ^bb5
    ! CHECK:   br ^bb10
    ! CHECK: ^bb9:  // pred: ^bb7
    ! CHECK:   fir.call @_FortranAioBeginExternalListOutput
    ! CHECK:   br ^bb10
    ! CHECK: ^bb10:  // 3 preds: ^bb6, ^bb8, ^bb9
    select case (char(ichar('0')+n))
      case ('1')
        print*, n, 'c:case 1'
      case ('2')
      ! print*, n, 'c:case 2'
      case default
        print*, n, 'c:case default'
    end select
    ! CHECK:   return
  end subroutine


  ! CHECK-LABEL: func @_QPswhere
  subroutine swhere(num)
    implicit none

    integer, intent(in) :: num
    real, dimension(1) :: array

    array = 0.0

    select case (num)
    ! CHECK: ^bb1:  // pred: ^bb0
    case (1)
      where (array >= 0.0)
        array = 42
      end where
    ! CHECK: cf.br ^bb3
    ! CHECK: ^bb2:  // pred: ^bb0
    case default
      array = -1
    end select
    ! CHECK: cf.br ^bb3
    ! CHECK: ^bb3:  // 2 preds: ^bb1, ^bb2
    print*, array(1)
  end subroutine swhere

  ! CHECK-LABEL: func @_QPsforall
  subroutine sforall(num)
    implicit none

    integer, intent(in) :: num
    real, dimension(1) :: array

    array = 0.0

    select case (num)
    ! CHECK: ^bb1:  // pred: ^bb0
    case (1)
      where (array >= 0.0)
        array = 42
      end where
    ! CHECK: cf.br ^bb3
    ! CHECK: ^bb2:  // pred: ^bb0
    case default
      array = -1
    end select
    ! CHECK: cf.br ^bb3
    ! CHECK: ^bb3:  // 2 preds: ^bb1, ^bb2
    print*, array(1)
  end subroutine sforall

  ! CHECK-LABEL: main
  program p
    integer sinteger, v(10)

    n = -10
    do j = 1, 4
      do k = 1, 10
        n = n + 1
        v(k) = sinteger(n)
      enddo
      ! expected output:  1 1 1 1 1 1 1 1 1 1
      !                   1 2 3 4 4 6 7 7 7 7
      !                   7 7 7 7 7 0 0 0 0 0
      !                   7 7 7 7 7 7 7 7 7 7
      print*, v
    enddo

    print*
    call slogical(.false.)    ! expected output:  0 1 0 3 1 1 3 1
    call slogical(.true.)     ! expected output:  0 0 2 3 2 3 2 2

    print*
    call scharacter('aa')     ! expected output: 10
    call scharacter('d')      ! expected output: 10
    call scharacter('f')      ! expected output: -1
    call scharacter('ff')     ! expected output: 20
    call scharacter('fff')    ! expected output: 20
    call scharacter('ffff')   ! expected output: 20
    call scharacter('fffff')  ! expected output: -1
    call scharacter('jj')     ! expected output: -1
    call scharacter('m')      ! expected output: 30
    call scharacter('q')      ! expected output: -1
    call scharacter('qq')     ! expected output: 40
    call scharacter('qqq')    ! expected output: -1
    call scharacter('vv')     ! expected output: -1
    call scharacter('xx')     ! expected output: 50
    call scharacter('zz')     ! expected output: 50

    print*
    call scharacter1('99 ')   ! expected output:  4
    call scharacter1('88 ')   ! expected output:  4
    call scharacter1('77 ')   ! expected output:  4
    call scharacter1('66 ')   ! expected output:  4
    call scharacter1('55 ')   ! expected output:  4
    call scharacter1('44 ')   ! expected output:  4
    call scharacter1('33 ')   ! expected output:  3
    call scharacter1('22 ')   ! expected output:  2
    call scharacter1('11 ')   ! expected output:  1
    call scharacter1('00 ')   ! expected output:  0
    call scharacter1('.  ')   ! expected output:  0
    call scharacter1('   ')   ! expected output:  0
 
    print*
    call scharacter2('99 ')   ! expected output:  9 -2
    call scharacter2('22 ')   ! expected output:  9 -2
    call scharacter2('.  ')   ! expected output:  9 -2
    call scharacter2('   ')   ! expected output:  9 -2

    print*
    call sempty(0)            ! expected output: 0 i:case default 0; c:case default
    call sempty(1)            ! expected output: 1 i:case 1; 1 c:case 1
    call sempty(2)            ! no output
    call sempty(3)            ! expected output: 3 i:case default; 3 c:case default

    print*
    call swhere(1)            ! expected output: 42.
    call sforall(1)           ! expected output: 42.
  end
