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
(W)     send.ugm (1|M0)          r6       r5  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     add (1|M0)               r7.0<1>:ud    r5.0<0;1,0>:ud    0x40:ud              {I@1}
        sync.nop                             null                             {$0.dst}
(W)     send.ugm (1|M0)          r8       r7  null:0  0xFF000000            0x6219D500           {A@1,$1} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     add (1|M0)               r9.0<1>:ud    r5.0<0;1,0>:ud    0x80:ud              {I@2}
        sync.nop                             null                             {$1.dst}
(W)     send.ugm (1|M0)          r5       r9  null:0  0xFF000000            0x6219D500           {A@1,$2} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
        sync.nop                             null                             {$2.src}
        mov (16|M0)              r9.0<1>:f     0.0:f                               {A@1}
        mov (16|M0)              r10.0<1>:f    0.0:f
        mov (16|M0)              r11.0<1>:f    0.0:f
        mov (16|M0)              r12.0<1>:f    0.0:f
        mov (16|M0)              r13.0<1>:f    0.0:f
        mov (16|M0)              r14.0<1>:f    0.0:f
        mov (16|M0)              r15.0<1>:f    0.0:f
        mov (16|M0)              r16.0<1>:f    0.0:f
        sync.nop                             null                             {$2.dst}
(W)     mov (1|M0)               r5.0<1>:q     r0.1<0;1,0>:ud
(W)     mov (1|M0)               r7.0<1>:q     r0.6<0;1,0>:ud
(W)     mov (1|M0)               r17.0<1>:q    6:q
(W)     shl (1|M0)               r18.0<1>:q    r5.0<0;1,0>:q     r17.0<0;1,0>:uw  {I@1}
(W)     mov (1|M0)               r5.0<1>:q     6:q
(W)     shl (1|M0)               r17.0<1>:q    r7.0<0;1,0>:q     r5.0<0;1,0>:uw   {I@1}
(W)     mov (1|M0)               r5.0<1>:ud    r1.0<0;1,0>:uw
(W)     shr (1|M0)               r7.0<1>:ud    r5.0<0;1,0>:ud    0x4:ud              {I@1}
(W)     mov (1|M0)               r5.0<1>:q     r7.0<0;1,0>:ud                   {I@1}
(W)     mov (1|M0)               r7.0<1>:q     3:q
(W)     and (1|M0)               r19.0<1>:q    r5.0<0;1,0>:q     r7.0<0;1,0>:q    {I@1}
(W)     mov (1|M0)               r7.0<1>:q     2:q
(W)     shr (1|M0)               r20.0<1>:q    r5.0<0;1,0>:q     r7.0<0;1,0>:uw   {I@1}
(W)     mov (1|M0)               r5.0<1>:q     3:q
(W)     and (1|M0)               r7.0<1>:q     r20.0<0;1,0>:q    r5.0<0;1,0>:q    {I@1}
(W)     mov (1|M0)               r5.0<1>:q     4:q
(W)     shl (1|M0)               r20.0<1>:q    r19.0<0;1,0>:q    r5.0<0;1,0>:uw   {I@1}
(W)     mov (1|M0)               r5.0<1>:q     63:q
(W)     and (1|M0)               r21.0<1>:q    r20.0<0;1,0>:q    r5.0<0;1,0>:q    {I@1}
(W)     add (1|M0)               r5.0<1>:q     r21.0<0;1,0>:q    r17.0<0;1,0>:q   {I@1}
(W)     mov (1|M0)               r17.0<1>:ud   r5.0<0;1,0>:q                    {I@1}
(W)     mov (1|M0)               r5.0<1>:q     4:q
(W)     shl (1|M0)               r20.0<1>:q    r7.0<0;1,0>:q     r5.0<0;1,0>:uw   {I@1}
(W)     mov (1|M0)               r5.0<1>:q     5:q
(W)     shl (1|M0)               r21.0<1>:q    r19.0<0;1,0>:q    r5.0<0;1,0>:uw   {I@1}
(W)     mov (1|M0)               r5.0<1>:q     63:q
(W)     and (1|M0)               r19.0<1>:q    r20.0<0;1,0>:q    r5.0<0;1,0>:q    {I@1}
(W)     mov (1|M0)               r5.0<1>:q     31:q
(W)     and (1|M0)               r20.0<1>:q    r21.0<0;1,0>:q    r5.0<0;1,0>:q    {I@1}
(W)     add (1|M0)               r5.0<1>:q     r19.0<0;1,0>:q    r18.0<0;1,0>:q   {I@3}
(W)     mov (1|M0)               r18.0<1>:q    8:q
(W)     add (1|M0)               r19.0<1>:q    r5.0<0;1,0>:q     r18.0<0;1,0>:q   {I@1}
(W)     mov (1|M0)               r18.0<1>:q    5:q
(W)     shl (1|M0)               r21.0<1>:q    r7.0<0;1,0>:q     r18.0<0;1,0>:uw  {I@1}
(W)     mov (1|M0)               r7.0<1>:q     31:q
(W)     and (1|M0)               r18.0<1>:q    r21.0<0;1,0>:q    r7.0<0;1,0>:q    {I@1}
(W)     mov (1|M0)               r7.0<1>:q     0:q
(W)     mov (1|M0)               r21.0<1>:ud   r5.0<0;1,0>:q
(W)     mov (1|M0)               r22.0<1>:q    r6.0<0;1,0>:q
(W)     mov (16|M0)              r23.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r23.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r23.2<1>:ud   0x7F:ud
(W)     mov (1|M0)               r23.3<1>:ud   0x7F:ud
(W)     mov (1|M0)               r23.4<1>:ud   0x7F:ud
        mov (2|M0)               r23.0<1>:ud   r22.0<1;1,0>:ud                  {I@6}
        mov (1|M0)               r23.6<1>:ud   r21.0<1;1,0>:ud
(W)     mov (16|M0)              r21.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r21.0<1>:ud   r19.0<0;1,0>:q
(W)     mov (16|M0)              r24.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r24.7<1>:ud   0x70F:ud
(W)     mov (1|M0)               r24.2<1>:ud   0x7F:ud
(W)     mov (1|M0)               r24.3<1>:ud   0x7F:ud
(W)     mov (1|M0)               r24.4<1>:ud   0x7F:ud
        mov (2|M0)               r24.0<1>:ud   r22.0<1;1,0>:ud
        mov (1|M0)               r24.6<1>:ud   r21.0<1;1,0>:ud                  {I@7}
(W)     mov (16|M0)              r21.0<1>:ud   0x0:ud
(W)     mov (1|M0)               r21.0<1>:q    r6.7<0;1,0>:q
(W)     mov (16|M0)              r6.0<1>:ud    0x0:ud
(W)     mov (1|M0)               r6.7<1>:ud    0xF0F:ud
(W)     mov (1|M0)               r6.2<1>:ud    0xFF:ud
(W)     mov (1|M0)               r6.3<1>:ud    0x3F:ud
(W)     mov (1|M0)               r6.4<1>:ud    0xFF:ud
        mov (2|M0)               r6.0<1>:ud    r21.0<1;1,0>:ud                  {I@6}
        mov (1|M0)               r6.5<1>:ud    r17.0<1;1,0>:ud
(W)     mov (16|M0)              r21.0<1>:ud   0x0:ud
(W)     mov (32|M0)              r25.0<1>:ud   r9.0<1;1,0>:ud                   {A@7}
(W)     mov (32|M0)              r27.0<1>:ud   r11.0<1;1,0>:ud                  {A@5}
(W)     mov (32|M0)              r29.0<1>:ud   r13.0<1;1,0>:ud                  {A@3}
(W)     mov (32|M0)              r31.0<1>:ud   r15.0<1;1,0>:ud                  {A@1}
L1616:
(W)     add (1|M0)               r21.0<1>:q    r20.0<0;1,0>:q    r7.0<0;1,0>:q    {I@3}
(W)     mov (1|M0)               r22.0<1>:ud   r21.0<0;1,0>:q                   {I@1}
        mov (16|M0)              r33.0<1>:ud   r23.0<1;1,0>:ud
        mov (1|M0)               r33.5<1>:ud   r22.0<1;1,0>:ud                  {I@2}
        sync.nop                             null                             {$9.src}
(W)     send.ugm (1|M0)          r34      r33  null:0  0x0            0x02400203           {I@1,$3} // wr:1+0, rd:4; load_block2d.ugm.d16.a64
(W)     mov (1|M0)               r38.0<1>:q    16:q
(W)     add (1|M0)               r39.0<1>:q    r21.0<0;1,0>:q    r38.0<0;1,0>:q   {I@1}
(W)     mov (1|M0)               r21.0<1>:ud   r39.0<0;1,0>:q                   {I@1}
        mov (16|M0)              r38.0<1>:ud   r23.0<1;1,0>:ud
        mov (1|M0)               r38.5<1>:ud   r21.0<1;1,0>:ud                  {I@2}
(W)     send.ugm (1|M0)          r39      r38  null:0  0x0            0x02400203           {A@1,$4} // wr:1+0, rd:4; load_block2d.ugm.d16.a64
        mov (16|M0)              r43.0<1>:ud   r24.0<1;1,0>:ud
        mov (1|M0)               r43.5<1>:ud   r22.0<1;1,0>:ud
        sync.nop                             null                             {$10.src}
(W)     send.ugm (1|M0)          r44      r43  null:0  0x0            0x02400203           {I@1,$5} // wr:1+0, rd:4; load_block2d.ugm.d16.a64
        mov (16|M0)              r22.0<1>:ud   r24.0<1;1,0>:ud
        mov (1|M0)               r22.5<1>:ud   r21.0<1;1,0>:ud                  {I@6}
(W)     send.ugm (1|M0)          r48      r22  null:0  0x0            0x02400203           {I@1,$6} // wr:1+0, rd:4; load_block2d.ugm.d16.a64
(W)     add (1|M0)               r21.0<1>:q    r18.0<0;1,0>:q    r7.0<0;1,0>:q
(W)     mov (1|M0)               r52.0<1>:ud   r21.0<0;1,0>:q                   {I@1}
        mov (16|M0)              r53.0<1>:ud   r6.0<1;1,0>:ud
        mov (1|M0)               r53.6<1>:ud   r52.0<1;1,0>:ud                  {I@2}
(W)     send.ugm (1|M0)          r54      r53  null:0  0x0            0x02800283           {I@1,$7} // wr:1+0, rd:8; load_block2d.ugm.d16v.a64
(W)     mov (1|M0)               r52.0<1>:q    16:q
(W)     add (1|M0)               r62.0<1>:q    r21.0<0;1,0>:q    r52.0<0;1,0>:q   {I@1}
(W)     mov (1|M0)               r21.0<1>:ud   r62.0<0;1,0>:q                   {I@1}
        mov (16|M0)              r52.0<1>:ud   r6.0<1;1,0>:ud
        mov (1|M0)               r52.6<1>:ud   r21.0<1;1,0>:ud                  {I@2}
(W)     send.ugm (1|M0)          r62      r52  null:0  0x0            0x02800283           {A@1,$8} // wr:1+0, rd:8; load_block2d.ugm.d16v.a64
        sync.nop                             null                             {A@1}
        sync.allwr                           ($3,$7,$9)
        dpas.8x8 (16|M0)         r9:f          r9:f              r54:hf            r34.0:hf         {$9}
        sync.nop                             null                             {A@1}
        sync.allwr                           ($4,$8)
        dpas.8x8 (16|M0)         r9:f          r9:f              r62:hf            r39.0:hf         {$9}
        sync.allwr                           ($5,$10)
        dpas.8x8 (16|M0)         r25:f         r25:f             r54:hf            r44.0:hf         {$10}
        sync.nop                             null                             {I@4}
        sync.nop                             null                             {$6.dst}
        dpas.8x8 (16|M0)         r25:f         r25:f             r62:hf            r48.0:hf         {$10}
(W)     mov (1|M0)               r21.0<1>:q    32:q
(W)     add (1|M0)               r7.0<1>:q     r7.0<1;1,0>:q     r21.0<0;1,0>:q   {I@1}
(W)     mov (1|M0)               r21.0<1>:q    64:q
        cmp (1|M0)    (lt)f0.0   null<1>:q     r7.0<1;1,0>:d     r21.0<0;1,0>:d   {I@1}
(W&f0.0) jmpi                                L1616
L2352:
(W)     mov (1|M0)               r6.0<1>:ud    r5.0<0;1,0>:q
(W)     mov (1|M0)               r5.0<1>:q     r8.6<0;1,0>:q
(W)     mov (16|M0)              r7.0<1>:ud    0x0:ud
(W)     mov (1|M0)               r7.7<1>:ud    0x70F:ud
(W)     mov (1|M0)               r7.2<1>:ud    0x1FF:ud
(W)     mov (1|M0)               r7.3<1>:ud    0x7F:ud
(W)     mov (1|M0)               r7.4<1>:ud    0x1FF:ud
        mov (2|M0)               r7.0<1>:ud    r5.0<1;1,0>:ud                   {I@6}
        mov (1|M0)               r7.5<1>:ud    r17.0<1;1,0>:ud
(W)     mov (16|M0)              r5.0<1>:ud    0x0:ud
        mov (16|M0)              r5.0<1>:ud    r7.0<1;1,0>:ud                   {I@2}
        mov (1|M0)               r5.6<1>:ud    r6.0<1;1,0>:ud
        sync.nop                             null                             {$9.dst}
(W)     send.ugm (1|M0)          null     r5  r9:8  0x0            0x02000407           {A@1,$11} // wr:1+8, rd:0; store_block2d.ugm.d32.a64
(W)     mov (1|M0)               r7.6<1>:ud    r19.0<0;1,0>:q
        sync.nop                             null                             {$10.dst}
        sync.nop                             null                             {$11.src}
(W)     send.ugm (1|M0)          null     r7  r25:8  0x0            0x02000407           {I@1,$12} // wr:1+8, rd:0; store_block2d.ugm.d32.a64
        sync.nop                             null                             {$12.src}
(W)     send.gtwy (1|M0)         null     r0  null:0  0x0            0x02000010           {EOT,$13} // wr:1+0, rd:0; end of thread
L2672:
