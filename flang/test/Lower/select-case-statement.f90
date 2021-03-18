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
  end
