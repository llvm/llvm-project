; We manually create these here if we're not linking against
; the CRT which would usually provide these.

target triple = "x86_64-pc-windows-msvc"

%IMAGE_TLS_DIRECTORY64 = type {
    i64,    ; StartAddressOfRawData
    i64,    ; EndAddressOfRawData
    i64,    ; AddressOfIndex
    i64,    ; AddressOfCallBacks
    i32,    ; SizeOfZeroFill
    i32     ; Characteristics
}

@_tls_start = global i8 zeroinitializer, section ".tls"
@_tls_end = global i8 zeroinitializer, section ".tls$ZZZ"
@_tls_index = global i64 0

@_tls_used = global %IMAGE_TLS_DIRECTORY64 {
    i64 ptrtoint (ptr @_tls_start to i64),
    i64 ptrtoint (ptr @_tls_end to i64),
    i64 ptrtoint (ptr @_tls_index to i64),
    i64 0,
    i32 0,
    i32 0
}, section ".rdata$T"