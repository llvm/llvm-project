L0:
(W)     mov (8|M0)               r2.0<1>:ud    r1.0<1;1,0>:ud
(W)     and (1|M0)               r5.0<1>:ud    r0.0<0;1,0>:ud    0xFFFFFFC0:ud
(W)     and (1|M0)               r7.0<1>:uw    r0.4<0;1,0>:uw    0xFF:uw
(W)     add (1|M0)               r6.0<1>:ud    r5.0<0;1,0>:ud    0x20:ud              {I@2}
(W)     mul (1|M0)               acc0.0<1>:ud  r7.0<0;1,0>:uw    0x40:uw              {I@2}
(W)     mov (1|M0)               r8.0<1>:ud    acc0.0<0;1,0>:ud                 {I@1}
(W)     add (1|M0)               r9.0<1>:ud    r6.0<0;1,0>:ud    r8.0<0;1,0>:ud   {I@1}
(W)     send.ugm (1|M0)          r1       r9  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
        sync.nop                             null                             {$0.dst}
        sync.nop                             null
        sync.nop                             null
        sync.nop                             null
        sync.nop                             null
        sync.allwr                           null
(W)     and (1|M0)               r5.0<1>:ud    r0.0<0;1,0>:ud    0xFFFFFFC0:ud
(W)     mov (1|M0)               r6.0<1>:q     r0.1<0;1,0>:ud
(W)     mov (1|M0)               r7.0<1>:q     r0.6<0;1,0>:ud
(W)     mov (1|M0)               r8.0<1>:q     6:q
(W)     mov (1|M0)               r9.0<1>:q     6:q
(W)     mov (1|M0)               r10.0<1>:ud   r1.0<0;1,0>:uw
(W)     mov (1|M0)               r11.0<1>:q    8:q
(W)     mov (16|M0)              r12.0<1>:ud   0x0:ud
(W)     send.ugm (1|M0)          r12      r5  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     add (1|M0)               r13.0<1>:ud   r5.0<0;1,0>:ud    0x40:ud
(W)     add (1|M0)               r14.0<1>:ud   r5.0<0;1,0>:ud    0x80:ud
(W)     shl (1|M0)               r15.0<1>:q    r6.0<0;1,0>:q     r8.0<0;1,0>:uw   {I@7}
(W)     shl (1|M0)               r6.0<1>:q     r7.0<0;1,0>:q     r9.0<0;1,0>:uw   {I@7}
(W)     shr (1|M0)               r7.0<1>:ud    r10.0<0;1,0>:ud   0x4:ud              {I@7}
(W)     mov (16|M0)              r8.0<1>:ud    0x0:ud
        sync.nop                             null                             {$0.dst}
(W)     send.ugm (1|M0)          r8       r13  null:0  0xFF000000            0x6219D500           {A@1,$1} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     mov (16|M0)              r5.0<1>:ud    0x0:ud
        sync.nop                             null                             {$1.dst}
(W)     send.ugm (1|M0)          r5       r14  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
        sync.nop                             null                             {$0.dst}
(W)     and (1|M0)               r5.0<1>:ud    r7.0<0;1,0>:ud    0x1:ud              {I@3}
(W)     shr (1|M0)               r9.0<1>:ud    r7.0<0;1,0>:ud    0x1:ud              {I@4}
(W)     and (1|M0)               r10.0<1>:ud   r7.0<0;1,0>:ud    0x3:ud              {I@5}
(W)     shr (1|M0)               r13.0<1>:ud   r7.0<0;1,0>:ud    0x2:ud              {I@6}
(W)     mov (16|M0)              r7.0<1>:ud    0x0:ud
        mov (16|M0)              r16.0<1>:f    0.0:f
(W)     mov (16|M0)              r7.0<1>:ud    0x0:ud
        mov (16|M0)              r17.0<1>:f    0.0:f
(W)     and (1|M0)               r7.0<1>:ud    r9.0<0;1,0>:ud    0x7:ud              {I@5}
        mov (16|M0)              r18.0<1>:f    0.0:f
(W)     shl (1|M0)               r9.0<1>:ud    r5.0<0;1,0>:ud    0x4:ud              {I@7}
        mov (16|M0)              r19.0<1>:f    0.0:f
(W)     and (1|M0)               r5.0<1>:ud    r13.0<0;1,0>:ud   0x3:ud              {I@5}
        mov (16|M0)              r20.0<1>:f    0.0:f
(W)     shl (1|M0)               r13.0<1>:ud   r10.0<0;1,0>:ud   0x4:ud              {I@7}
        mov (16|M0)              r21.0<1>:f    0.0:f
(W)     shl (1|M0)               r14.0<1>:ud   r10.0<0;1,0>:ud   0x5:ud
        mov (16|M0)              r22.0<1>:f    0.0:f
(W)     shl (1|M0)               r10.0<1>:ud   r7.0<0;1,0>:ud    0x3:ud              {I@5}
        mov (16|M0)              r23.0<1>:f    0.0:f
(W)     mov (16|M0)              r7.0<1>:ud    0x0:ud
(W)     and (1|M0)               r7.0<1>:ud    r10.0<0;1,0>:ud   0x3F:ud              {I@2}
(W)     mov (1|M0)               r10.0<1>:q    r12.0<0;1,0>:q
(W)     shl (1|M0)               r24.0<1>:ud   r5.0<0;1,0>:ud    0x3:ud              {I@7}
(W)     and (1|M0)               r25.0<1>:ud   r13.0<0;1,0>:ud   0x3F:ud              {I@7}
(W)     mov (1|M0)               r13.0<1>:q    r12.7<0;1,0>:q
(W)     shl (1|M0)               r12.0<1>:ud   r5.0<0;1,0>:ud    0x4:ud
(W)     shl (1|M0)               r26.0<1>:ud   r5.0<0;1,0>:ud    0x5:ud
(W)     mov (1|M0)               r5.0<1>:q     r7.0<0;1,0>:ud                   {I@7}
(W)     and (1|M0)               r7.0<1>:ud    r9.0<0;1,0>:ud    0x1F:ud
(W)     add (1|M0)               r9.0<1>:q     r5.0<0;1,0>:q     r15.0<0;1,0>:q   {I@2}
(W)     mov (1|M0)               r5.0<1>:q     r25.0<0;1,0>:ud                  {I@7}
(W)     and (1|M0)               r25.0<1>:ud   r12.0<0;1,0>:ud   0x3F:ud              {I@6}
(W)     mov (1|M0)               r12.0<1>:ud   r9.0<0;1,0>:q                    {I@3}
(W)     mov (16|M0)              r9.0<1>:ud    0x0:ud
(W)     mov (1|M0)               r9.7<1>:ud    0x70F:ud
(W)     mov (1|M0)               r9.2<1>:ud    0x7F:ud
(W)     mov (1|M0)               r9.3<1>:ud    0x7F:ud
(W)     mov (1|M0)               r9.4<1>:ud    0x7F:ud
        mov (2|M0)               r9.0<1>:ud    r10.0<1;1,0>:ud
        mov (1|M0)               r9.5<1>:ud    r7.0<1;1,0>:ud
(W)     add (1|M0)               r27.0<1>:q    r5.0<0;1,0>:q     r6.0<0;1,0>:q
        mov (1|M0)               r9.6<1>:ud    r12.0<1;1,0>:ud
(W)     mov (1|M0)               r5.0<1>:q     r25.0<0;1,0>:ud
(W)     send.ugm (1|M0)          null     r9  null:0  0x0            0x02080203           {I@2,$0} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
(W)     and (1|M0)               r6.0<1>:ud    r24.0<0;1,0>:ud   0x1F:ud
(W)     mov (1|M0)               r24.0<1>:ud   r27.0<0;1,0>:q                   {I@4}
(W)     mov (16|M0)              r25.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r25.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r25.2<1>:ud   0xFF:ud
(W)     mov (1|M0)               r25.3<1>:ud   0x3F:ud
(W)     mov (1|M0)               r25.4<1>:ud   0xFF:ud
        mov (2|M0)               r25.0<1>:ud   r13.0<1;1,0>:ud
        mov (1|M0)               r25.5<1>:ud   r24.0<1;1,0>:ud                  {I@7}
        mov (1|M0)               r25.6<1>:ud   r6.0<1;1,0>:ud
(W)     send.ugm (1|M0)          null     r25  null:0  0x0            0x02080203           {I@1,$1} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
(W)     and (1|M0)               r27.0<1>:ud   r14.0<0;1,0>:ud   0x1F:ud
(W)     add (1|M0)               r14.0<1>:q    r5.0<0;1,0>:q     r15.0<0;1,0>:q
(W)     add (1|M0)               r5.0<1>:q     r14.0<0;1,0>:q    r11.0<0;1,0>:q   {I@1}
(W)     and (1|M0)               r11.0<1>:ud   r26.0<0;1,0>:ud   0x1F:ud
(W)     mov (1|M0)               r15.0<1>:ud   0x0:ud
(W)     mov (16|M0)              r26.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r28.0<1>:ud   0x70F:ud
(W)     mov (1|M0)               r26.2<1>:ud   0xFF:ud
(W)     mov (1|M0)               r26.3<1>:ud   0x3F:ud
(W)     mov (1|M0)               r26.4<1>:ud   0xFF:ud
        mov (2|M0)               r26.0<1>:ud   r13.0<1;1,0>:ud
        mov (1|M0)               r26.5<1>:ud   r24.0<1;1,0>:ud
(W)     mov (16|M0)              r13.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r13.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r13.2<1>:ud   0x7F:ud
(W)     mov (1|M0)               r13.3<1>:ud   0x7F:ud
(W)     mov (1|M0)               r13.4<1>:ud   0x7F:ud
        mov (2|M0)               r13.0<1>:ud   r10.0<1;1,0>:ud
(W)     mov (1|M0)               r10.0<1>:ud   r14.0<0;1,0>:q
(W)     mov (1|M0)               r29.0<1>:ud   0xF0F:ud
(W)     mov (32|M0)              r30.0<1>:ud   r16.0<1;1,0>:ud                  {A@7}
(W)     mov (32|M0)              r32.0<1>:ud   r18.0<1;1,0>:ud                  {A@5}
(W)     mov (32|M0)              r34.0<1>:ud   r20.0<1;1,0>:ud                  {A@3}
(W)     mov (32|M0)              r36.0<1>:ud   r22.0<1;1,0>:ud                  {A@1}
L1840:
        sync.nop                             null                             {$4.src}
(W)     add (1|M0)               r38.0<1>:ud   r15.0<0;1,0>:ud   0x20:ud              {I@2}
(W)     add (1|M0)               r39.0<1>:ud   r27.0<0;1,0>:ud   r15.0<0;1,0>:ud  {I@3}
(W)     add (1|M0)               r40.0<1>:ud   r11.0<0;1,0>:ud   r15.0<0;1,0>:ud  {I@4}
(W)     add (1|M0)               r41.0<1>:ud   r6.0<0;1,0>:ud    r38.0<0;1,0>:ud  {I@3}
        sync.nop                             null                             {$3.src}
        mov (16|M0)              r42.0<1>:ud   r26.0<1;1,0>:ud
        mov (1|M0)               r42.6<1>:ud   r41.0<1;1,0>:ud                  {I@2}
        mov (1|M0)               r42.7<1>:ud   r28.0<1;1,0>:ud
(W)     add (1|M0)               r41.0<1>:ud   r7.0<0;1,0>:ud    r38.0<0;1,0>:ud  {I@7}
(W)     send.ugm (1|M0)          null     r42  null:0  0x0            0x02080203           {I@2,$3} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
        mov (16|M0)              r38.0<1>:ud   r13.0<1;1,0>:ud
        mov (1|M0)               r38.5<1>:ud   r41.0<1;1,0>:ud                  {I@2}
        mov (1|M0)               r38.6<1>:ud   r12.0<1;1,0>:ud
(W)     send.ugm (1|M0)          null     r38  null:0  0x0            0x02080203           {I@1,$4} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
        mov (16|M0)              r41.0<1>:ud   r13.0<1;1,0>:ud
        mov (1|M0)               r41.5<1>:ud   r39.0<1;1,0>:ud
        mov (1|M0)               r41.6<1>:ud   r10.0<1;1,0>:ud
        sync.nop                             null                             {$9.src}
(W)     send.ugm (1|M0)          r43      r41  null:0  0x0            0x02400203           {I@1,$2} // wr:1+0, rd:4; load_block2d.ugm.d16.a64
        sync.nop                             null                             {$9.src}
(W)     send.ugm (1|M0)          r47      r41  null:0  0x10000            0x02400203           {I@1,$5} // wr:1+0, rd:4; load_block2d.ugm.d16.a64.flat[A+(0x10,0x0)]
        sync.nop                             null                             {$10.src}
(W)     send.ugm (1|M0)          r51      r41  null:0  0x2000000            0x02400203           {I@1,$6} // wr:1+0, rd:4; load_block2d.ugm.d16.a64.flat[A+(0x0,0x8)]
        sync.nop                             null                             {$10.src}
(W)     send.ugm (1|M0)          r55      r41  null:0  0x2010000            0x02400203           {I@1,$7} // wr:1+0, rd:4; load_block2d.ugm.d16.a64.flat[A+(0x10,0x8)]
        sync.nop                             null                             {$0.src}
        mov (16|M0)              r9.0<1>:ud    r26.0<1;1,0>:ud
        mov (1|M0)               r9.6<1>:ud    r40.0<1;1,0>:ud
        mov (1|M0)               r9.7<1>:ud    r29.0<1;1,0>:ud
(W)     send.ugm (1|M0)          r59      r9  null:0  0x0            0x02800283           {I@1,$0} // wr:1+0, rd:8; load_block2d.ugm.d16v.a64
(W)     send.ugm (1|M0)          r67      r9  null:0  0x4000000            0x02800283           {I@1,$8} // wr:1+0, rd:8; load_block2d.ugm.d16v.a64.flat[A+(0x0,0x10)]
        sync.nop                             null                             {A@1}
        sync.allwr                           ($0,$2,$6,$9,$10)
        dpas.8x8 (16|M0)         r16:f         r16:f             r59:hf            r43.0:hf         {$9}
        dpas.8x8 (16|M0)         r30:f         r30:f             r59:hf            r51.0:hf         {$10}
        sync.nop                             null                             {A@1}
        sync.allwr                           ($5,$7,$8)
        dpas.8x8 (16|M0)         r16:f         r16:f             r67:hf            r47.0:hf         {$9}
        dpas.8x8 (16|M0)         r30:f         r30:f             r67:hf            r55.0:hf         {$10}
(W)     add (1|M0)               r15.0<1>:ud   r15.0<1;1,0>:ud   0x20:ud
        cmp (1|M0)    (lt)f0.0   null<1>:ud    r15.0<1;1,0>:d    64:d               {I@1}
(W&f0.0) jmpi                                L1840
L2528:
(W)     mov (1|M0)               r6.0<1>:ud    r14.0<0;1,0>:q
(W)     mov (1|M0)               r7.0<1>:q     r8.6<0;1,0>:q
(W)     mov (16|M0)              r8.0<1>:ud    0x0:ud
(W)     mov (1|M0)               r8.7<1>:ud    0x70F:ud
(W)     mov (1|M0)               r8.2<1>:ud    0x1FF:ud
(W)     mov (1|M0)               r8.3<1>:ud    0x7F:ud
(W)     mov (1|M0)               r8.4<1>:ud    0x1FF:ud
        mov (2|M0)               r8.0<1>:ud    r7.0<1;1,0>:ud                   {I@6}
        mov (1|M0)               r8.5<1>:ud    r24.0<1;1,0>:ud
(W)     mov (16|M0)              r7.0<1>:ud    0x0:ud
        mov (16|M0)              r7.0<1>:ud    r8.0<1;1,0>:ud                   {I@2}
        mov (1|M0)               r7.6<1>:ud    r6.0<1;1,0>:ud
        sync.nop                             null                             {$9.dst}
(W)     send.ugm (1|M0)          null     r7  r16:8  0x0            0x02000407           {A@1,$0} // wr:1+8, rd:0; store_block2d.ugm.d32.a64
(W)     mov (1|M0)               r8.6<1>:ud    r5.0<0;1,0>:q
        sync.nop                             null                             {$0.src}
        sync.nop                             null                             {$10.dst}
(W)     send.ugm (1|M0)          null     r8  r30:8  0x0            0x02000407           {I@1,$2} // wr:1+8, rd:0; store_block2d.ugm.d32.a64
        sync.nop                             null                             {$1.src}
        sync.nop                             null                             {$2.src}
        sync.nop                             null                             {$3.src}
        sync.nop                             null                             {$4.src}
(W)     send.gtwy (1|M0)         null     r0  null:0  0x0            0x02000010           {EOT,$0} // wr:1+0, rd:0; end of thread
L2896:
