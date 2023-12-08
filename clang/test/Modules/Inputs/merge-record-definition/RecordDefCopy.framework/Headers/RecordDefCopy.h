// It is important to have a definition *after* non-definition declaration.
typedef struct _Buffer Buffer;
struct _Buffer {
  int a;
  int b;
  int c;
};

typedef struct _AnonymousStruct AnonymousStruct;
struct _AnonymousStruct {
  struct {
    int x;
    int y;
  };
};

typedef union _UnionRecord UnionRecord;
union _UnionRecord {
  int u: 2;
  int v: 4;
};
