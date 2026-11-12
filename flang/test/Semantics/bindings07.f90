! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
module ma
  type a
   contains
    procedure, private, nopass :: tbp => sub_a
    generic :: gen => tbp
  end type
  type, extends(a) :: aa
   contains
    procedure, private, nopass :: tbp => sub_aa
  end type
  type, extends(aa) :: aaa
   contains
    procedure, public, nopass :: tbp => sub_aaa
  end type
 contains
  subroutine sub_a(w)
    character*(*), intent(in) :: w
    print *, w, ' -> a'
  end
  subroutine sub_aa(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aa'
  end
  subroutine sub_aaa(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aaa'
  end
  subroutine mono1
    type(a) :: xa
    type(aa) :: xaa
    call xa%tbp('type(a) tbp')
    call xaa%tbp('type(aa) tbp')
  end
  subroutine pa(x, w)
    class(a), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(a) ' // w // ' tbp')
    call x%gen('class(a) ' // w // ' gen')
  end
  subroutine pta1
    call pa(a(), 'a')
    call pa(aa(), 'aa')
  end
  subroutine paa(x, w)
    class(aa), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aa) ' // w // ' tbp')
    call x%gen('class(aa) ' // w // ' gen')
  end
  subroutine ptaa1
    call paa(aa(), 'aa')
  end
  subroutine paaa(x, w)
    class(aaa), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aaa) ' // w // ' tbp')
    call x%gen('class(aaa) ' // w // ' gen')
  end
  subroutine ptaaa1
    call paaa(aaa(), 'aaa')
  end
end

module mb
  use ma
  type, extends(a) :: ab
   contains
    procedure, public, nopass :: tbp => sub_ab
  end type
  type, extends(aa) :: aab
   contains
    procedure, public, nopass :: tbp => sub_aab
  end type
  type, extends(aaa) :: aaab
   contains
    procedure, public, nopass :: tbp => sub_aaab
  end type
  type, extends(ab) :: aba
   contains
    procedure, public, nopass :: tbp => sub_aba
  end type
  type, extends(aab) :: aaba
   contains
    procedure, public, nopass :: tbp => sub_aaba
  end type
  type, extends(aaab) :: aaaba
   contains
    procedure, public, nopass :: tbp => sub_aaaba
  end type
 contains
  subroutine sub_ab(w)
    character*(*), intent(in) :: w
    print *, w, ' -> ab'
  end
  subroutine sub_aab(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aab'
  end
  subroutine sub_aaab(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aaab'
  end
  subroutine sub_aba(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aba'
  end
  subroutine sub_aaba(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aaba'
  end
  subroutine sub_aaaba(w)
    character*(*), intent(in) :: w
    print *, w, ' -> aaaba'
  end
end

module t
  use mb
 contains
  subroutine mono2
    type(a) :: xa
    type(aa) :: xaa
    type(aaa) :: xaaa
    type(ab) :: xab
    type(aab) :: xaab
    type(aaab) :: xaaab
    type(aba) :: xaba
    type(aaba) :: xaaba
    type(aaaba) :: xaaaba
    call xa%gen('type(a) gen')
    call xaa%gen('type(aa) gen')
    call xaaa%tbp('type(aaa) tbp')
    call xaaa%gen('type(aaa) gen')
    call xab%tbp('type(ab) tbp')
    call xab%gen('type(ab) gen')
    call xaab%tbp('type(aab) tbp')
    call xaab%gen('type(aab) gen')
    call xaaab%tbp('type(aaab) tbp')
    call xaaab%gen('type(aaab) gen')
    call xaba%tbp('type(aba) tbp')
    call xaba%gen('type(aba) gen')
    call xaaba%tbp('type(aaba) tbp')
    call xaaba%gen('type(aaba) gen')
    call xaaaba%tbp('type(aaaba) tbp')
    call xaaaba%gen('type(aaaba) gen')
  end
  subroutine pta2
    call pa(a(), 'a')
    call pa(aa(), 'aa')
    call pa(aaa(), 'aaa')
    call pa(ab(), 'ab')
    call pa(aab(), 'aab')
    call pa(aaab(), 'aaab')
    call pa(aba(), 'aba')
    call pa(aaba(), 'aaba')
    call pa(aaaba(), 'aaaba')
  end
  subroutine ptaa2
    call paa(aa(), 'aa')
    call paa(aaa(), 'aaa')
    call paa(aab(), 'aab')
    call paa(aaab(), 'aaab')
    call paa(aaba(), 'aaba')
    call paa(aaaba(), 'aaaba')
  end
  subroutine ptaaa2
    call paaa(aaa(), 'aaa')
    call paaa(aaab(), 'aaab')
    call paaa(aaaba(), 'aaaba')
  end
  subroutine pab(x, w)
    class(ab), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(ab) ' // w // ' tbp')
    call x%gen('class(ab) ' // w // ' gen')
  end
  subroutine ptab
    call pab(ab(), 'ab')
    call pab(aba(), 'aba')
  end
  subroutine paab(x, w)
    class(aab), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aab) ' // w // ' tbp')
    call x%gen('class(aab) ' // w // ' gen')
  end
  subroutine ptaab
    call pa(aab(), 'aab')
    call pa(aaba(), 'aaba')
  end
  subroutine paaab(x, w)
    class(aaab), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aaab) ' // w // ' tbp')
    call x%gen('class(aaab) ' // w // ' gen')
  end
  subroutine ptaaab
    call pa(aaab(), 'aaab')
    call pa(aaaba(), 'aaaba')
  end
  subroutine paba(x, w)
    class(aba), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aba) ' // w // ' tbp')
    call x%gen('class(aba) ' // w // ' gen')
  end
  subroutine ptaba
    call paba(aba(), 'aba')
  end
  subroutine paaba(x, w)
    class(aaba), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aaba) ' // w // ' tbp')
    call x%gen('class(aaba) ' // w // ' gen')
  end
  subroutine ptaaba
    call paaba(aaba(), 'aaba')
  end
  subroutine paaaba(x, w)
    class(aaaba), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp('class(aaaba) ' // w // ' tbp')
    call x%gen('class(aaaba) ' // w // ' gen')
  end
  subroutine ptaaaba
    call pa(aaaba(), 'aaaba')
  end
end

program main
  use t
  call mono1
  call mono2
  call pta1
  call ptaa1
  call ptaaa1
  call pta2
  call ptaa2
  call ptaaa2
  call ptab
  call ptaab
  call ptaaab
  call ptaba
  call ptaaba
  call ptaaaba
end

!CHECK: .v.a, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=sub_a,name=.n.tbp)]
!CHECK: .v.aa, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=sub_aa,name=.n.tbp)]
!CHECK: .v.aaa, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=sub_aaa,name=.n.tbp)]
!CHECK: .v.aaab, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=sub_aaab,name=.n.tbp)]
!CHECK: .v.aaaba, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=sub_aaaba,name=.n.tbp)]
!CHECK: .v.aab, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=sub_aa,name=.n.tbp),binding(proc=sub_aab,name=.n.tbp)]
!CHECK: .v.aaba, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=sub_aa,name=.n.tbp),binding(proc=sub_aaba,name=.n.tbp)]
!CHECK: .v.ab, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=sub_a,name=.n.tbp),binding(proc=sub_ab,name=.n.tbp)]
!CHECK: .v.aba, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=sub_a,name=.n.tbp),binding(proc=sub_aba,name=.n.tbp)]
!CHECK: tbp, NOPASS, PUBLIC: ProcBinding => sub_ab numPrivatesNotOverridden: 1
!CHECK: tbp, NOPASS, PUBLIC: ProcBinding => sub_aab numPrivatesNotOverridden: 1
!CHECK: tbp, NOPASS, PUBLIC: ProcBinding => sub_aba numPrivatesNotOverridden: 1
!CHECK: tbp, NOPASS, PUBLIC: ProcBinding => sub_aaba numPrivatesNotOverridden: 1
