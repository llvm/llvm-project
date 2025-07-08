; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define hidden spir_func noundef nofpclass(nan inf) half @_Z17test_refract_halfDhDhDh(half noundef nofpclass(nan inf) %I, half noundef nofpclass(nan inf) %N, half noundef nofpclass(nan inf) %ETA) local_unnamed_addr #0 {
entry:
  %mul.i = fmul reassoc nnan ninf nsz arcp afn half %N, %I
  %mul1.i = fmul reassoc nnan ninf nsz arcp afn half %ETA, %ETA
  %mul2.i = fmul reassoc nnan ninf nsz arcp afn half %mul.i, %mul.i
  %sub.i = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, %mul2.i
  %mul3.i = fmul reassoc nnan ninf nsz arcp afn half %mul1.i, %sub.i
  %sub4.i = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, %mul3.i
  %mul5.i = fmul reassoc nnan ninf nsz arcp afn half %ETA, %I
  %mul6.i = fmul reassoc nnan ninf nsz arcp afn half %ETA, %mul.i
  %0 = tail call reassoc nnan ninf nsz arcp afn half @llvm.sqrt.f16(half %sub4.i)
  %add.i = fadd reassoc nnan ninf nsz arcp afn half %0, %mul6.i
  %mul7.i = fmul reassoc nnan ninf nsz arcp afn half %add.i, %N
  %sub8.i = fsub reassoc nnan ninf nsz arcp afn half %mul5.i, %mul7.i
  %cmp.i = fcmp reassoc nnan ninf nsz arcp afn olt half %sub4.i, 0xH0000
  %hlsl.select.i = select reassoc nnan ninf nsz arcp afn i1 %cmp.i, half 0xH0000, half %sub8.i
  ret half %hlsl.select.i
}