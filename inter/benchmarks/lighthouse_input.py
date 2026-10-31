from __future__ import annotations


def specialize_inter_source(
    source_text: str, size: int, reduction_size: int
) -> str:
    replacements = {
        "%18 = llvm.mlir.constant(512 : i32) : i32": (
            f"%18 = llvm.mlir.constant({size * 4} : i32) : i32\n"
            f"    %matrix_size = llvm.mlir.constant({size} : i32) : i32\n"
            "    %reduction_size = "
            f"llvm.mlir.constant({reduction_size} : i32) : i32\n"
            "    %reduction_bytes = "
            f"llvm.mlir.constant({reduction_size * 2} : i32) : i32\n"
            "    %reduction_bound = "
            f"llvm.mlir.constant({reduction_size} : i64) : i64"
        ),
        "%15 = llvm.mlir.constant(256 : i32) : i32": (
            f"%15 = llvm.mlir.constant({size * 2} : i32) : i32"
        ),
        "(%45, %7, %7, %7, %47)": (
            "(%45, %reduction_bytes, %matrix_size, %reduction_bytes, %47)"
        ),
        "(%45, %7, %7, %7, %88)": (
            "(%45, %reduction_bytes, %matrix_size, %reduction_bytes, %88)"
        ),
        "(%45, %7, %7, %7, %93, %94)": (
            "(%45, %reduction_bytes, %matrix_size, %reduction_bytes, %93, %94)"
        ),
        "(%45, %7, %7, %7, %100, %101)": (
            "(%45, %reduction_bytes, %matrix_size, %reduction_bytes, %100, %101)"
        ),
        "(%45, %7, %7, %7, %106, %107)": (
            "(%45, %reduction_bytes, %matrix_size, %reduction_bytes, %106, %107)"
        ),
        "(%45, %7, %7, %7, %111, %112)": (
            "(%45, %reduction_bytes, %matrix_size, %reduction_bytes, %111, %112)"
        ),
        "(%65, %15, %5, %15, %67)": (
            "(%65, %15, %reduction_size, %15, %67)"
        ),
        "(%65, %15, %5, %15, %84)": (
            "(%65, %15, %reduction_size, %15, %84)"
        ),
        "(%65, %15, %5, %15, %118, %119)": (
            "(%65, %15, %reduction_size, %15, %118, %119)"
        ),
        "(%65, %15, %5, %15, %125, %126)": (
            "(%65, %15, %reduction_size, %15, %125, %126)"
        ),
        '%79 = llvm.icmp "slt" %76, %2 : i64': (
            '%79 = llvm.icmp "slt" %76, %reduction_bound : i64'
        ),
        "(%149, %18, %7, %18, %152, %153)": (
            "(%149, %18, %matrix_size, %18, %152, %153)"
        ),
        "(%149, %18, %7, %18, %157, %158)": (
            "(%149, %18, %matrix_size, %18, %157, %158)"
        ),
    }
    for frozen, replacement in replacements.items():
        if source_text.count(frozen) != 1:
            raise ValueError(f"frozen Inter input no longer has one {frozen!r}")
        source_text = source_text.replace(frozen, replacement)
    return source_text


def drop_loop_prefetches(source_text: str) -> str:
    in_loop = False
    removed = 0
    lines = []
    for line in source_text.splitlines():
        in_loop |= line.startswith("  ^bb1")
        if (
            in_loop
            and line.lstrip().startswith("llvm.call")
            and "intel_sub_group_2d_block_prefetch" in line
        ):
            removed += 1
            continue
        lines.append(line)
    if removed != 2:
        raise ValueError(f"expected two loop prefetches, found {removed}")
    return "\n".join(lines) + "\n"
