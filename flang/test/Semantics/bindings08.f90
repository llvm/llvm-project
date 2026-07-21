! RUN: %python %S/test_errors.py %s %flang_fc1

module m
    type, abstract :: parent
        contains
        procedure(ip), deferred :: set
    end type
    abstract interface
        subroutine ip(x)
            import parent
            class(parent) :: x
        end subroutine
    end interface
    type, public, extends(parent), abstract :: child
    end type
    type, extends(child) :: grandchild
        contains
        !ERROR: Passed-object dummy arguments of type-bound procedure 'set' and its override must correspond by name and position
        procedure :: set
    end type
    contains
    subroutine set(y)
        class(grandchild) :: y
    end subroutine
end module
