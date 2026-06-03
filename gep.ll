define ptr @gep_i64(ptr %base, i64 %idx) {
  %p = getelementptr inbounds i64, ptr %base, i64 %idx
  ret ptr %p
}
