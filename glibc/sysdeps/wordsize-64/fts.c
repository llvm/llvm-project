#define fts64_open __rename_fts64_open
#define fts64_close __rename_fts64_close
#define fts64_read __rename_fts64_read
#define fts64_set __rename_fts64_set
#define fts64_children __rename_fts64_children

#include "../../io/fts.c"

#undef fts64_open
#undef fts64_close
#undef fts64_read
#undef fts64_set
#undef fts64_children

weak_alias (fts_open, fts64_open)
weak_alias (fts_close, fts64_close)
weak_alias (fts_read, fts64_read)
weak_alias (fts_set, fts64_set)
weak_alias (fts_children, fts64_children)
