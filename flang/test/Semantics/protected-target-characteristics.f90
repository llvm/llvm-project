! RUN: %flang_fc1 -fsyntax-only -Wproc-dummy-arg-shapes -Werror %s

module pt_chars
  interface
    module subroutine sep_iface_pt(a)
      integer, pointer, protected_target :: a
    end
    module subroutine sep_iface_plain(a)
      integer, pointer :: a
    end
    module function sep_iface_pt_result() result(r)
      integer, pointer, protected_target :: r
    end
    module function sep_iface_plain_result() result(r)
      integer, pointer :: r
    end
  end interface
contains
  function plain_result() result(r)
    integer, target, save :: x = 1
    integer, pointer :: r
    r => x
  end

  function protected_result() result(r)
    integer, target, save :: x = 1
    integer, pointer, protected_target :: r
    r => x
  end

  subroutine plain_dummy(a)
    integer, pointer :: a
    a = 99
  end

  subroutine protected_dummy(a)
    integer, pointer, protected_target :: a
  end

  subroutine takes_proc_result(proc)
    interface
      function iface_pt_result() result(r)
        integer, pointer, protected_target :: r
      end
    end interface
    procedure(iface_pt_result) :: proc
  end

  subroutine takes_proc_plain_result(proc)
    interface
      function iface_plain_result() result(r)
        integer, pointer :: r
      end
    end interface
    procedure(iface_plain_result) :: proc
  end

  subroutine takes_proc_dummy(proc)
    interface
      subroutine iface_pt_dummy(a)
        integer, pointer, protected_target :: a
      end
    end interface
    procedure(iface_pt_dummy) :: proc
  end

  subroutine takes_proc_plain_dummy(proc)
    interface
      subroutine iface_plain_dummy(a)
        integer, pointer :: a
      end
    end interface
    procedure(iface_plain_dummy) :: proc
  end

  subroutine test
    abstract interface
      function iface_pt_result() result(r)
        integer, pointer, protected_target :: r
      end
      function iface_plain_result() result(r)
        integer, pointer :: r
      end
      subroutine iface_pt_dummy(a)
        integer, pointer, protected_target :: a
      end
      subroutine iface_plain_dummy(a)
        integer, pointer :: a
      end
    end interface
    procedure(iface_pt_result), pointer :: pfr_pt
    procedure(iface_plain_result), pointer :: pfr_plain
    procedure(iface_pt_dummy), pointer :: ppd_pt
    procedure(iface_plain_dummy), pointer :: ppd_plain
    pfr_pt => plain_result
    pfr_plain => protected_result
    ppd_pt => plain_dummy
    ppd_plain => protected_dummy
    call takes_proc_result(plain_result)
    call takes_proc_result(protected_result)
    call takes_proc_plain_result(plain_result)
    call takes_proc_plain_result(protected_result)
    call takes_proc_dummy(plain_dummy)
    call takes_proc_dummy(protected_dummy)
    call takes_proc_plain_dummy(plain_dummy)
    call takes_proc_plain_dummy(protected_dummy)
  end
end

submodule(pt_chars) pt_chars_sub
contains
  module subroutine sep_iface_pt(a)
    integer, pointer :: a
  end

  module subroutine sep_iface_plain(a)
    integer, pointer, protected_target :: a
  end

  module function sep_iface_pt_result() result(r)
    integer, pointer :: r
    integer, target, save :: x = 1
    r => x
  end

  module function sep_iface_plain_result() result(r)
    integer, pointer, protected_target :: r
    integer, target, save :: x = 1
    r => x
  end
end
