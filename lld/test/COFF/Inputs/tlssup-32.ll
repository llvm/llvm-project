; We manually create these here if we're not linking against
; the CRT which would usually provide these.

target triple = "i686-pc-windows-msvc"

%IMAGE_TLS_DIRECTORY32 = type {
    i32,    ; StartAddressOfRawData
    i32,    ; EndAddressOfRawData
    i32,    ; AddressOfIndex
    i32,    ; AddressOfCallBacks
    i32,    ; SizeOfZeroFill
    i32     ; Characteristics
}

@_tls_start = global i8 zeroinitializer, section ".tls"
@_tls_end = global i8 zeroinitializer, section ".tls$ZZZ"
@_tls_index = global i32 0

@_tls_used = global %IMAGE_TLS_DIRECTORY32 {
    i32 ptrtoint (i8* @_tls_start to i32),
    i32 ptrtoint (i8* @_tls_end to i32),
    i32 ptrtoint (i32* @_tls_index to i32),
    i32 0,
    i32 0,
    i32 0
}, section ".rdata$T"

; MSVC target uses a direct offset (0x58) for x86-64 but expects
; __tls_array to hold the offset (0x2C) on x86.
module asm ".global __tls_array"
module asm "__tls_array = 44"