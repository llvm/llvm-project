; Module with a target-abi flag but no target datalayout line.
target triple = "riscv32-unknown-unknown-elf"

define i32 @f() #0 {
  ret i32 0
}

attributes #0 = { "target-features"="+32bit,+e" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"target-abi", !"ilp32e"}
