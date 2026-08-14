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
        mov (16|M0)              r16.0<1>:f    0.0:f
        mov (16|M0)              r17.0<1>:f    0.0:f
(W)     and (1|M0)               r7.0<1>:ud    r9.0<0;1,0>:ud    0x7:ud              {I@3}
        mov (16|M0)              r18.0<1>:f    0.0:f
(W)     shl (1|M0)               r9.0<1>:ud    r5.0<0;1,0>:ud    0x4:ud              {I@5}
        mov (16|M0)              r19.0<1>:f    0.0:f
(W)     and (1|M0)               r5.0<1>:ud    r13.0<0;1,0>:ud   0x3:ud              {I@3}
        mov (16|M0)              r20.0<1>:f    0.0:f
(W)     shl (1|M0)               r13.0<1>:ud   r10.0<0;1,0>:ud   0x4:ud              {I@5}
        mov (16|M0)              r21.0<1>:f    0.0:f
(W)     shl (1|M0)               r14.0<1>:ud   r10.0<0;1,0>:ud   0x5:ud              {I@6}
        mov (16|M0)              r22.0<1>:f    0.0:f
(W)     shl (1|M0)               r10.0<1>:ud   r7.0<0;1,0>:ud    0x3:ud              {I@5}
        mov (16|M0)              r23.0<1>:f    0.0:f
(W)     and (1|M0)               r7.0<1>:ud    r9.0<0;1,0>:ud    0x1F:ud              {I@5}
(W)     mov (1|M0)               r9.0<1>:q     r12.0<0;1,0>:q
(W)     shl (1|M0)               r24.0<1>:ud   r5.0<0;1,0>:ud    0x3:ud              {I@6}
(W)     shl (1|M0)               r25.0<1>:ud   r5.0<0;1,0>:ud    0x4:ud              {I@7}
(W)     and (1|M0)               r26.0<1>:ud   r13.0<0;1,0>:ud   0x3F:ud              {I@7}
(W)     shl (1|M0)               r13.0<1>:ud   r5.0<0;1,0>:ud    0x5:ud
(W)     and (1|M0)               r5.0<1>:ud    r10.0<0;1,0>:ud   0x3F:ud              {I@7}
(W)     mov (1|M0)               r10.0<1>:q    r5.0<0;1,0>:ud                   {I@1}
(W)     mov (1|M0)               r5.0<1>:q     r26.0<0;1,0>:ud                  {I@4}
(W)     and (1|M0)               r26.0<1>:ud   r25.0<0;1,0>:ud   0x3F:ud              {I@6}
(W)     add (1|M0)               r25.0<1>:q    r10.0<0;1,0>:q    r15.0<0;1,0>:q   {I@3}
(W)     add (1|M0)               r10.0<1>:q    r5.0<0;1,0>:q     r6.0<0;1,0>:q    {I@3}
(W)     mov (1|M0)               r5.0<1>:q     r26.0<0;1,0>:ud                  {I@3}
(W)     mov (1|M0)               r6.0<1>:ud    r25.0<0;1,0>:q                   {I@3}
(W)     mov (16|M0)              r25.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r25.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r25.2<1>:ud   0x7F:ud
(W)     mov (1|M0)               r25.3<1>:ud   0x7F:ud
(W)     mov (1|M0)               r25.4<1>:ud   0x7F:ud
        mov (2|M0)               r25.0<1>:ud   r9.0<1;1,0>:ud
        mov (1|M0)               r25.5<1>:ud   r7.0<1;1,0>:ud
        mov (1|M0)               r25.6<1>:ud   r6.0<1;1,0>:ud
(W)     send.ugm (1|M0)          null     r25  null:0  0x0            0x02080203           {I@1,$0} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
(W)     and (1|M0)               r6.0<1>:ud    r24.0<0;1,0>:ud   0x1F:ud
(W)     mov (1|M0)               r7.0<1>:ud    r10.0<0;1,0>:q
(W)     mov (1|M0)               r10.0<1>:q    r12.7<0;1,0>:q
(W)     mov (16|M0)              r12.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r12.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r12.2<1>:ud   0xFF:ud
(W)     mov (1|M0)               r12.3<1>:ud   0x3F:ud
(W)     mov (1|M0)               r12.4<1>:ud   0xFF:ud
        mov (2|M0)               r12.0<1>:ud   r10.0<1;1,0>:ud                  {I@6}
        mov (1|M0)               r12.5<1>:ud   r7.0<1;1,0>:ud
        mov (1|M0)               r12.6<1>:ud   r6.0<1;1,0>:ud
(W)     send.ugm (1|M0)          null     r12  null:0  0x0            0x02080203           {I@1,$1} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
(W)     and (1|M0)               r24.0<1>:ud   r14.0<0;1,0>:ud   0x1F:ud
(W)     add (1|M0)               r14.0<1>:q    r5.0<0;1,0>:q     r15.0<0;1,0>:q
(W)     add (1|M0)               r5.0<1>:q     r14.0<0;1,0>:q    r11.0<0;1,0>:q   {I@1}
(W)     and (1|M0)               r11.0<1>:ud   r13.0<0;1,0>:ud   0x1F:ud
(W)     mov (1|M0)               r13.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r15.0<1>:ud   r14.0<0;1,0>:q                   {I@4}
(W)     mov (16|M0)              r26.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r26.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r26.2<1>:ud   0x7F:ud
(W)     mov (1|M0)               r26.3<1>:ud   0x7F:ud
(W)     mov (1|M0)               r26.4<1>:ud   0x7F:ud
        mov (2|M0)               r26.0<1>:ud   r9.0<1;1,0>:ud
        mov (1|M0)               r26.6<1>:ud   r15.0<1;1,0>:ud                  {I@7}
(W)     mov (16|M0)              r9.0<1>:ud    0x0:ud
(W)     mov (1|M0)               r9.7<1>:ud    0xF0F:ud
(W)     mov (1|M0)               r9.2<1>:ud    0xFF:ud
(W)     mov (1|M0)               r9.3<1>:ud    0x3F:ud
(W)     mov (1|M0)               r9.4<1>:ud    0xFF:ud
        mov (2|M0)               r9.0<1>:ud    r10.0<1;1,0>:ud
        mov (1|M0)               r9.5<1>:ud    r7.0<1;1,0>:ud
(W)     mov (16|M0)              r15.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r27.0<1>:ud   0x70F:ud
(W)     mov (1|M0)               r28.0<1>:ud   0xFF:ud
(W)     mov (1|M0)               r29.0<1>:ud   0x3F:ud
(W)     mov (1|M0)               r30.0<1>:ud   0xFF:ud
(W)     mov (32|M0)              r31.0<1>:ud   r16.0<1;1,0>:ud                  {A@7}
(W)     mov (32|M0)              r33.0<1>:ud   r18.0<1;1,0>:ud                  {A@5}
(W)     mov (32|M0)              r35.0<1>:ud   r20.0<1;1,0>:ud                  {A@3}
(W)     mov (32|M0)              r37.0<1>:ud   r22.0<1;1,0>:ud                  {A@1}
L1872:
        sync.nop                             null                             {$9.src}
(W)     add (1|M0)               r39.0<1>:ud   r13.0<0;1,0>:ud   0x20:ud              {I@2}
(W)     mov (1|M0)               r40.0<1>:q    64:q
(W)     mov (1|M0)               r41.0<1>:q    r39.0<0;1,0>:ud                  {I@2}
        cmp (1|M0)    (lt)f0.0   null<1>:q     r41.0<0;1,0>:d    r40.0<0;1,0>:d   {I@1}
(W&~f0.0) jmpi                               L2144
L1968:
(W)     add (1|M0)               r40.0<1>:ud   r6.0<0;1,0>:ud    r39.0<0;1,0>:ud  {I@4}
        mov (16|M0)              r39.0<1>:ud   r15.0<1;1,0>:ud
        mov (2|M0)               r39.0<1>:ud   r10.0<1;1,0>:ud
        mov (1|M0)               r39.2<1>:ud   r28.0<1;1,0>:ud
        mov (1|M0)               r39.3<1>:ud   r29.0<1;1,0>:ud
        mov (1|M0)               r39.4<1>:ud   r30.0<1;1,0>:ud
        mov (1|M0)               r39.5<1>:ud   r7.0<1;1,0>:ud
        mov (1|M0)               r39.6<1>:ud   r40.0<1;1,0>:ud                  {I@7}
        mov (1|M0)               r39.7<1>:ud   r27.0<1;1,0>:ud
(W)     send.ugm (1|M0)          null     r39  null:0  0x0            0x02080203           {I@1,$9} // wr:1+0, rd:0; load_block2d.ugm.d16.a64.ca.ca
(W)     jmpi                                 L2144
L2144:
(W)     add (1|M0)               r40.0<1>:ud   r24.0<0;1,0>:ud   r13.0<0;1,0>:ud  {I@6}
        mov (16|M0)              r41.0<1>:ud   r26.0<1;1,0>:ud
        mov (1|M0)               r41.5<1>:ud   r40.0<1;1,0>:ud                  {I@2}
(W)     add (1|M0)               r40.0<1>:ud   r11.0<0;1,0>:ud   r13.0<0;1,0>:ud
        sync.nop                             null                             {$7.src}
(W)     send.ugm (1|M0)          r42      r41  null:0  0x0            0x02400203           {I@2,$2} // wr:1+0, rd:4; load_block2d.ugm.d16.a64
        sync.nop                             null                             {$7.src}
(W)     send.ugm (1|M0)          r46      r41  null:0  0x10000            0x02400203           {I@2,$3} // wr:1+0, rd:4; load_block2d.ugm.d16.a64.flat[A+(0x10,0x0)]
        sync.nop                             null                             {$8.src}
(W)     send.ugm (1|M0)          r50      r41  null:0  0x2000000            0x02400203           {I@2,$4} // wr:1+0, rd:4; load_block2d.ugm.d16.a64.flat[A+(0x0,0x8)]
        sync.nop                             null                             {$8.src}
(W)     send.ugm (1|M0)          r54      r41  null:0  0x2010000            0x02400203           {I@2,$5} // wr:1+0, rd:4; load_block2d.ugm.d16.a64.flat[A+(0x10,0x8)]
        sync.nop                             null                             {$1.src}
        mov (16|M0)              r12.0<1>:ud   r9.0<1;1,0>:ud
        mov (1|M0)               r12.6<1>:ud   r40.0<1;1,0>:ud                  {I@2}
(W)     send.ugm (1|M0)          r58      r12  null:0  0x0            0x02800283           {I@1,$1} // wr:1+0, rd:8; load_block2d.ugm.d16v.a64
(W)     send.ugm (1|M0)          r66      r12  null:0  0x4000000            0x02800283           {I@1,$6} // wr:1+0, rd:8; load_block2d.ugm.d16v.a64.flat[A+(0x0,0x10)]
        sync.nop                             null                             {A@1}
        sync.allwr                           ($1,$2,$4,$7,$8)
        dpas.8x8 (16|M0)         r16:f         r16:f             r58:hf            r42.0:hf         {$7}
        dpas.8x8 (16|M0)         r31:f         r31:f             r58:hf            r50.0:hf         {$8}
        sync.nop                             null                             {A@1}
        sync.allwr                           ($3,$5,$6)
        dpas.8x8 (16|M0)         r16:f         r16:f             r66:hf            r46.0:hf         {$7}
        dpas.8x8 (16|M0)         r31:f         r31:f             r66:hf            r54.0:hf         {$8}
(W)     add (1|M0)               r13.0<1>:ud   r13.0<1;1,0>:ud   0x20:ud
        cmp (1|M0)    (lt)f0.0   null<1>:ud    r13.0<1;1,0>:d    64:d               {I@1}
(W&f0.0) jmpi                                L1872
L2592:
(W)     mov (1|M0)               r6.0<1>:ud    r14.0<0;1,0>:q
(W)     mov (1|M0)               r9.0<1>:q     r8.6<0;1,0>:q
(W)     mov (16|M0)              r8.0<1>:ud    0x0:ud
(W)     mov (1|M0)               r8.7<1>:ud    0x70F:ud
(W)     mov (1|M0)               r8.2<1>:ud    0x1FF:ud
(W)     mov (1|M0)               r8.3<1>:ud    0x7F:ud
(W)     mov (1|M0)               r8.4<1>:ud    0x1FF:ud
        mov (2|M0)               r8.0<1>:ud    r9.0<1;1,0>:ud                   {I@6}
        mov (1|M0)               r8.5<1>:ud    r7.0<1;1,0>:ud
(W)     mov (16|M0)              r7.0<1>:ud    0x0:ud
        mov (16|M0)              r7.0<1>:ud    r8.0<1;1,0>:ud                   {I@2}
        mov (1|M0)               r7.6<1>:ud    r6.0<1;1,0>:ud
        sync.nop                             null                             {$7.dst}
(W)     send.ugm (1|M0)          null     r7  r16:8  0x0            0x02000407           {A@1,$1} // wr:1+8, rd:0; store_block2d.ugm.d32.a64
(W)     mov (1|M0)               r8.6<1>:ud    r5.0<0;1,0>:q
        sync.nop                             null                             {$1.src}
        sync.nop                             null                             {$8.dst}
(W)     send.ugm (1|M0)          null     r8  r31:8  0x0            0x02000407           {I@1,$2} // wr:1+8, rd:0; store_block2d.ugm.d32.a64
        sync.nop                             null                             {$0.src}
        sync.nop                             null                             {$2.src}
        sync.nop                             null                             {$9.src}
(W)     send.gtwy (1|M0)         null     r0  null:0  0x0            0x02000010           {EOT,$1} // wr:1+0, rd:0; end of thread
L2944:
