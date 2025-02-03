void asdf_counted(int * buf, int len);
void asdf_sized(int * buf, int size);
void asdf_counted_n(int * buf, int len);
void asdf_sized_n(int * buf, int size);
void asdf_ended(int * buf, int * end);

void asdf_sized_mul(int * buf, int size, int count);
void asdf_counted_out(int ** buf, int * len);
void asdf_counted_const(int * buf);
void asdf_counted_nullable(int len, int * _Nullable buf);
void asdf_counted_noescape(int * buf, int len);
void asdf_counted_default_level(int * buf, int len);
void asdf_counted_redundant(int * __attribute__((__counted_by__(len))) buf, int len);

void asdf_ended_chained(int * buf, int * mid, int * end);
void asdf_ended_chained_reverse(int * buf, int * mid, int * end);
void asdf_ended_already_started(int * __attribute__((__ended_by__(mid))) buf, int * mid, int * end);
void asdf_ended_already_ended(int * buf, int * mid __attribute__((__ended_by__(end))), int * end);
void asdf_ended_redundant(int * __attribute__((__ended_by__(end))) buf, int * end);

void asdf_nterm(char * buf);
