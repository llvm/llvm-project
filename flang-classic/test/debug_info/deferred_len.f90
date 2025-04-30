!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !DILocalVariable(name: "defl_string"{{.*}}, type: ![[DERIVEDSTRING:[0-9]+]]
!CHECK: ![[DERIVEDSTRING]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[STRING:[0-9]+]]
!CHECK: ![[STRING]] = distinct !DIStringType(name: "character(*)", stringLength: !{{[0-9]+}}

program deferredlength

        character(len=100) :: buffer
        character(len=:), allocatable :: defl_string

        read(*,*) buffer
        defl_string = trim(buffer)
        print *,defl_string

end program deferredlength
