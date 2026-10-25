.386  

_text segment para 'CODE'  ; Start the code segment.       
    align 16
    public _false_header
    public _function_to_intercept

_false_header proc
cc:
    db 66h, 90h               ; Padding function to force offset to 0xCC
    nop dword ptr [eax+eax*1+10000000h] 
    nop dword ptr [eax+eax*1+10000000h]
    nop dword ptr [eax+eax*1+10000000h]
    nop dword ptr [eax+eax*1+10000000h]
    nop dword ptr [eax+eax*1+10000000h]
    nop dword ptr [eax+eax*1+10000000h]
    jmp cc                    ; Short jump with a 0xCC byte used as an offset
    int 3                     ; 4 bytes of 0xCC padding
    int 3
    int 3
    int 3
_false_header endp

_function_to_intercept proc
    mov edi, edi               ; Function to be overridden
    push ebp
    mov ebp, esp
    mov eax, 0 
    pop ebp
    ret           
_function_to_intercept endp

_text ends       

end