! RUN: bbc -emit-fir -o - %s | FileCheck %s

  !CHECK-LABEL: sinteger
  function sinteger(n)
    integer sinteger
    nn = -88
    ! CHECK-DAG: fir.select_case {{.*}} : i32
    ! CHECK-SAME: upper, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: point, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: #fir.point, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: #fir.interval, %c{{[0-9]+}}_i32, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: #fir.point, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: #fir.point, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: #fir.interval, %c{{[0-9]+}}_i32, %c{{[0-9]+}}_i32,
    ! CHECK-SAME: #fir.lower, %c{{[0-9]+}}_i32,
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
      ! CHECK: arith.cmpi eq, %{{[0-9]+}}, %false
      ! CHECK: cf.cond_br
      case (.false.)
        n2 = 1
    end select

    select case (L)
      ! CHECK: cf.cond_br
      case (.true.)
        n3 = 2
    end select

    select case (L)
      case default
        n4 = 3
    end select

    select case (L)
      ! CHECK: arith.cmpi eq, %{{[0-9]+}}, %false
      ! CHECK: cf.cond_br
      case (.false.)
        n5 = 1
      ! CHECK: cf.cond_br
      case (.true.)
        n5 = 2
    end select

    select case (L)
      ! CHECK: arith.cmpi eq, %{{[0-9]+}}, %false
      ! CHECK: cf.cond_br
      case (.false.)
        n6 = 1
      case default
        n6 = 3
    end select

    select case (L)
      ! CHECK: cf.cond_br
      case (.true.)
        n7 = 2
      case default
        n7 = 3
    end select

    select case (L)
      ! CHECK: arith.cmpi eq, %{{[0-9]+}}, %false
      ! CHECK: cf.cond_br
      case (.false.)
        n8 = 1
      ! CHECK: cf.cond_br
      case (.true.)
        n8 = 2
      ! CHECK-NOT: 888
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
      ! CHECK-NEXT: arith.cmpi sle, %{{[0-9]+}}, %c{{[0-9]+}}_i32 : i32
      ! CHECK-NEXT: cf.cond_br
      case (:'d')
        nn = 10
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: arith.cmpi sge, %{{[0-9]+}}, %c{{[0-9]+}}_i32 : i32
      ! CHECK-NEXT: cf.cond_br
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: arith.cmpi sle, %{{[0-9]+}}, %c{{[0-9]+}}_i32 : i32
      ! CHECK-NEXT: cf.cond_br
      case ('ff':'ffff')
        nn = 20
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: arith.cmpi eq, %{{[0-9]+}}, %c{{[0-9]+}}_i32 : i32
      ! CHECK-NEXT: cf.cond_br
      case ('m')
        nn = 30
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: arith.cmpi eq, %{{[0-9]+}}, %c{{[0-9]+}}_i32 : i32
      ! CHECK-NEXT: cf.cond_br
      case ('qq')
        nn = 40
      ! CHECK: CharacterCompareScalar1
      ! CHECK-NEXT: arith.cmpi sge, %{{[0-9]+}}, %c{{[0-9]+}}_i32 : i32
      ! CHECK-NEXT: cf.cond_br
      case ('x':)
        nn = 50
    end select
    print*, nn
  end

  ! CHECK-LABEL: func.func @_QPscharacter1
  subroutine scharacter1(s)
    character(len=3) :: s
    n = 0

    ! lge() is lowered to various loops and "if" statements that work with "00".
    ! It's not our goal to completely lge() lowering here.
    ! CHECK: fir.declare {{.*}} uniq_name = "_QQclX3030"}
    ! CHECK: fir.do_loop
    if (lge(s,'00')) then

      ! CHECK: fir.call @_FortranATrim

      ! All the strings in SELECT CASE
      ! CHECK: %[[STR11:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3131"}
      ! CHECK: %[[STR22:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3232"}
      ! CHECK: %[[STR33:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3333"}
      ! CHECK: %[[STR44:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3434"}
      ! CHECK: %[[STR55:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3535"}
      ! CHECK: %[[STR66:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3636"}
      ! CHECK: %[[STR77:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3737"}
      ! CHECK: %[[STR88:[0-9]+]] = fir.declare {{.*}} uniq_name = "_QQclX3838"}

      ! == '11'
      ! CHECK: %[[STR11_CONV:[0-9]+]] = fir.convert %[[STR11]]
      ! CHECK: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR11_CONV]]
      ! CHECK-NEXT: arith.cmpi eq
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb3, ^bb2
      select case(trim(s))
      case('11')
        n = 1

      case default
        continue

      ! == '22'
      ! CHECK-DAG: %[[STR22_CONV:[0-9]+]] = fir.convert %[[STR22]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR22_CONV]]
      ! CHECK-NEXT: arith.cmpi eq,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb5, ^bb4
      case('22')
        n = 2

      ! == '33'
      ! CHECK-DAG: %[[STR33_CONV:[0-9]+]] = fir.convert %[[STR33]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR33_CONV]]
      ! CHECK-NEXT: arith.cmpi eq,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb7, ^bb6
      case('33')
        n = 3

      ! >= '44'
      ! CHECK-DAG: %[[STR44_CONV:[0-9]+]] = fir.convert %[[STR44]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR44_CONV]]
      ! CHECK-NEXT: arith.cmpi sge,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb8, ^bb9
      ! <= '55'
      ! CHECK-DAG: %[[STR55_CONV:[0-9]+]] = fir.convert %[[STR55]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR55_CONV]]
      ! CHECK-NEXT: arith.cmpi sle,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb12, ^bb9
      ! >= '66'
      ! CHECK-DAG: %[[STR66_CONV:[0-9]+]] = fir.convert %[[STR66]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR66_CONV]]
      ! CHECK-NEXT: arith.cmpi sge,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb10, ^bb11
      ! <= '77'
      ! CHECK-DAG: %[[STR77_CONV:[0-9]+]] = fir.convert %[[STR77]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR77_CONV]]
      ! CHECK-NEXT: arith.cmpi sle,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb12, ^bb11
      ! >= '88'
      ! CHECK-DAG: %[[STR88_CONV:[0-9]+]] = fir.convert %[[STR88]]
      ! CHECK-NEXT: fir.call @_FortranACharacterCompareScalar1({{.*}}, %[[STR88_CONV]]
      ! CHECK-NEXT: arith.cmpi sge,{{.*}}, %c0_i32 : i32
      ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb12, ^bb13
      case('44':'55','66':'77','88':)
        n = 4
      end select
    end if
    print*, n
  end subroutine

  ! CHECK-LABEL: func @_QPscharacter2
  subroutine scharacter2(s)
    character(len=3) :: s
    ! CHECK: %[[N:[0-9]+]] = fir.declare {{.*}} {uniq_name = "_QFscharacter2En"}
    ! CHECK: fir.store %c-10_i32 to %[[N]] : !fir.ref<i32>
    n = -10
    ! CHECK: fir.call @_FortranATrim(
    select case(trim(s))
    case default
      ! CHECK: fir.store %c9_i32 to %[[N]] : !fir.ref<i32>
      n = 9
    end select

    ! CHECK: fir.call @_FortranAioBeginExternalListOutput(
    print*, n

    ! CHECK:  fir.store %c-2_i32 to %[[N]] : !fir.ref<i32>
    n = -2

    ! CHECK: fir.call @_FortranATrim(
    select case(trim(s))
    end select
    ! CHECK: fir.call @_FortranAioBeginExternalListOutput(
    print*, n
  end subroutine

  ! CHECK-LABEL: func @_QPsempty
  ! empty select case blocks
  subroutine sempty(n)
    !CHECK: fir.select_case {{.*}} : i32 [#fir.point, %c1_i32, ^bb1, #fir.point, %c2_i32, ^bb2, unit, ^bb3]
    select case (n)
      case (1)
        !CHECK: ^bb1:
        !CHECK: fir.call @_FortranAioBeginExternalListOutput(
        !CHECK: cf.br ^bb4
        print*, n, 'i:case 1'
      case (2)
        !CHECK: ^bb2:
        !CHECK-NEXT: cf.br ^bb4
      ! (empty) print*, n, 'i:case 2'
      case default
        print*, n, 'i:case default'
    end select
    select case (char(ichar('0')+n))
    ! CHECK: fir.call @_FortranACharacterCompareScalar1(
    ! CHECK-NEXT: arith.cmpi eq, {{.*}}, %c0_i32 : i32
    ! CHECK-NEXT: cf.cond_br
      case ('1')
        print*, n, 'c:case 1'
      case ('2')
    ! CHECK: fir.call @_FortranACharacterCompareScalar1(
    ! CHECK-NEXT: arith.cmpi eq, {{.*}}, %c0_i32 : i32
    ! CHECK-NEXT: cf.cond_br {{.*}}, ^bb8
      ! (empty) print*, n, 'c:case 2'
      case default
        print*, n, 'c:case default'
    end select
    ! CHECK: ^bb8:
    ! CHECK-NEXT: return
  end subroutine

  ! CHECK-LABEL: func @_QPsgoto
  ! select case with goto exit
  subroutine sgoto
    n = 0
    ! CHECK: cf.cond_br {{.*}}, ^bb2, ^bb8
    do i=1,8
      ! CHECK: ^bb2: 
      ! CHECK: fir.select_case %8 : i32 [#fir.upper, %c2_i32, ^bb3, #fir.lower, %c5_i32, ^bb4, unit, ^bb6]
      select case(i)
      case (:2)
        ! CHECK-DAG: arith.muli {{.*}}, %c10_i32 : i32
        n = i * 10
      case (5:)
        ! CHECK-DAG: arith.muli {{.*}}, %c1000_i32 : i32
        n = i * 1000
        ! CHECK-DAG: arith.cmpi sle, {{.*}}, %c6_i32 : i32
        ! CHECK-NEXT: cf.cond_br
        if (i <= 6) goto 9
        ! CHECK-DAG: arith.muli {{.*}}, %c10000_i32 : i32
        n = i * 10000
      case default
        ! CHECK-DAG: arith.muli {{.*}}, %c100_i32 : i32
        n = i * 100
  9   end select
      print*, n
    enddo
    ! CHECK: ^bb8:
    ! CHECK-NEXT: return
  end

  ! CHECK-LABEL: func @_QPswhere
  subroutine swhere(num)
    implicit none

    integer, intent(in) :: num
    real, dimension(1) :: array

    array = 0.0

    ! CHECK: fir.select_case {{.*}} : i32 [#fir.point, %c1_i32, ^bb1, unit, ^bb2]
    select case (num)
    case (1)
      ! CHECK: fir.do_loop
      where (array >= 0.0)
        array = 42
      end where
    case default
      array = -1
    end select
    ! CHECK: cf.br ^bb3
    print*, array(1)
  end subroutine swhere

  ! CHECK-LABEL: func @_QPsforall
  subroutine sforall(num)
    implicit none

    integer, intent(in) :: num
    real, dimension(1) :: array
    integer :: i

    array = 0.0

    ! CHECK: fir.select_case {{.*}} : i32 [#fir.point, %c1_i32, ^bb1, unit, ^bb2]
    select case (num)
    case (1)
      ! CHECK: fir.do_loop
      forall (i = 1:size(array)) array(i) = 42
    case default
      array = -1
    end select
    ! CHECK: cf.br ^bb3
    print*, array(1)
  end subroutine sforall

  ! CHECK-LABEL: func @_QPsnested
  subroutine snested(str)
    character(*), optional :: str
    integer :: num

    ! CHECK: fir.is_present
    if (present(str)) then
      ! CHECK: fir.call @_FortranATrim
      select case (trim(str))
        ! CHECK: fir.call @_FortranACharacterCompareScalar1
        ! CHECK-NEXT: arith.cmpi eq, {{.*}}, %c0_i32 : i32
        case ('a')
          ! CHECK-DAG: fir.store %c10_i32 to {{.*}} : !fir.ref<i32>
          num = 10
        case default
          ! CHECK-DAG: fir.store %c20_i32 to {{.*}} : !fir.ref<i32>
          num = 20
      end select
    else
      ! CHECK-DAG: fir.store %c30_i32 to {{.*}} : !fir.ref<i32>
      num = 30
    end if
  end subroutine snested

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
    call sgoto                ! expected output:  10 20 300 400 5000 6000 70000 80000

    print*
    call swhere(1)            ! expected output: 42.
    call sforall(1)           ! expected output: 42.
  end
