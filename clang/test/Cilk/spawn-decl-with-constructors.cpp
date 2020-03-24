// Test code generation for uses of _Cilk_spawn in variable declarations.
//
// Thanks to Dr. I-Ting Angelina Lee for contributing the original source code
// for this test case.
//
// RUN: %clang_cc1 %s -std=c++11 -triple x86_64-unknown-linux-gnu -fcilkplus -fexceptions -ftapir=none -S -emit-llvm -o - | FileCheck %s

template <typename T1, typename T2>
struct pair {
  T1 first;
  T2 second;
  pair(T1 first, T2 second) : first(first), second(second) {}

  template<class U1, class U2>
  pair(const pair<U1, U2>& __p)
      : first(__p.first), second(__p.second) { }
};

template <typename T1, typename T2>
pair<T1, T2> make_pair(T1 &&t1, T2 &&t2) {
  return pair<T1, T2>(t1, t2);
}

template <class T>
struct _seq {
  T* A;
  long n;
  _seq() {A = 0; n=0;}
  _seq(T* _A, long _n) : A(_A), n(_n) {}
  void del() {free(A);}
};

typedef long intT;

template <class floatT> class _point2d;

template <class _floatT> class _vect2d {
public: 
  typedef _floatT floatT;
  typedef _point2d<floatT> pointT;
  typedef _vect2d vectT;
  floatT x; floatT y;
  _vect2d(floatT xx,floatT yy) : x(xx),y(yy) {}
  _vect2d() {x=0;y=0;}
  _vect2d(pointT p);
  _vect2d(floatT* p) : x(p[0]), y(p[1]) {};
  vectT operator+(vectT op2) {return vectT(x + op2.x, y + op2.y);}
  vectT operator-(vectT op2) {return vectT(x - op2.x, y - op2.y);}
  pointT operator+(pointT op2);
  vectT operator*(floatT s) {return vectT(x * s, y * s);}
  vectT operator/(floatT s) {return vectT(x / s, y / s);}
  floatT operator[] (int i) {return (i==0) ? x : y;};
  floatT dot(vectT v) {return x * v.x + y * v.y;}
  floatT cross(vectT v) { return x*v.y - y*v.x; }  
  floatT maxDim() {return max(x,y);}
  floatT Length(void) { return sqrt(x*x+y*y);}
  static const int dim = 3;
};

template <class _floatT> class _point2d {
public: 
  typedef _floatT floatT;
  typedef _vect2d<floatT> vectT;
  typedef _point2d pointT;
  floatT x; floatT y; 
  int dimension() {return 2;}
  _point2d(floatT xx,floatT yy) : x(xx),y(yy) {}
  _point2d() {x=0;y=0;}
  _point2d(vectT v) : x(v.x),y(v.y) {};
  _point2d(floatT* p) : x(p[0]), y(p[1]) {};
  vectT operator-(pointT op2) {return vectT(x - op2.x, y - op2.y);}
  pointT operator+(vectT op2) {return pointT(x + op2.x, y + op2.y);}
  floatT operator[] (int i) {return (i==0) ? x : y;};
  pointT minCoords(pointT b) { return pointT(min(x,b.x),min(y,b.y)); }
  pointT maxCoords(pointT b) { return pointT(max(x,b.x),max(y,b.y)); }
  int quadrant(pointT center) {
    int index = 0;
    if (x > center.x) index += 1;
    if (y > center.y) index += 2;
    return index;
  }
  // returns a pointT offset by offset in one of 4 directions 
  // depending on dir (an integer from [0..3])
  pointT offsetPoint(int dir, floatT offset) {
    floatT xx = x + ((dir & 1) ? offset : -offset);
    floatT yy = y + ((dir & 2) ? offset : -offset);
    return pointT(xx,yy);
  }
  bool outOfBox(pointT pt, floatT hsize) { 
    return ((x - hsize > pt.x) || (x + hsize < pt.x) ||
	    (y - hsize > pt.y) || (y + hsize < pt.y));
  }
  static const int dim = 2;
};

typedef _point2d<double> point2d;

template<typename _FIter, typename _Compare>
pair<_FIter, _FIter>
minmax_element(_FIter, _FIter, _Compare);

pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > merge(pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmax_e1, pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmax_e2);

bool compare_x(const point2d &p1, const point2d &p2){
  if(p1.x < p2.x){
    return true;
  }else if (p1.x == p2.x){
    return p1.y < p2.y;
  }else{
    return false;
  }
}

bool compare_y(const point2d &p1, const point2d &p2){
  if(p1.y < p2.y){
    return true;
  }else if(p1.x == p2.x){
    return p1.x < p2.x;
  }else{
    return false;
  }
}
static const intT minmax_base = 2000;

double triArea(point2d a, point2d b, point2d c);
struct aboveLineP {
  point2d l, r;
  point2d* P;
  aboveLineP(point2d* _P, point2d &_l, point2d &_r) : P(_P), l(_l), r(_r) {}
  bool operator() (point2d &i) {return triArea(l, r, i) > 0.0;}
};

point2d *offset_helper(point2d * buf, long off){
  char *tmp = (char *)buf;
  tmp = tmp + off;
  return (point2d *)tmp;
}

intT quickHullP(point2d* P, point2d* Ptmp, intT n, point2d l, point2d r, intT depth);

template <class ET, class intT, class PRED>
intT wrapped_filter_new(ET* In, ET* Out, intT n, PRED p);

pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > find_minmax_xy(point2d *p,  intT n){
  if (n < minmax_base) {
    pair<point2d *, point2d *> minmax_ex = minmax_element(p, p+n, compare_x);
    pair<point2d *, point2d *> minmax_ey = minmax_element(p, p+n, compare_y);
    return make_pair(minmax_ex, minmax_ey);
  } else {
    pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmax_e1 = _Cilk_spawn find_minmax_xy(p, n/2);
    pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmax_e2 = find_minmax_xy(p+n/2, n - n/2);
    _Cilk_sync;
    return merge(minmax_e1, minmax_e2);
  }
}

// CHECK-LABEL: @_Z14find_minmax_xyP8_point2dIdEl(
// CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED]]:
// CHECK: invoke void @_Z14find_minmax_xyP8_point2dIdEl(%struct.pair* sret %minmax_e1,
// CHECK-NEXT: to label %[[AFTERINVOKE:.+]] unwind
// CHECK: [[AFTERINVOKE]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]

_seq<point2d> hullP(point2d* P, intT n, point2d *Ptmp) {
  intT num_pages = (n * sizeof(point2d));
  pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmaxxy1 = _Cilk_spawn find_minmax_xy(P, n/4);
  pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmaxxy2 = _Cilk_spawn find_minmax_xy(offset_helper(P, num_pages), n/4);
  pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmaxxy3 = _Cilk_spawn find_minmax_xy(offset_helper(P, num_pages*2), n/4);
  pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmaxxy4 = find_minmax_xy(offset_helper(P, num_pages*3), n - n/4*3);
  _Cilk_sync;

  pair<pair<point2d *, point2d *>, pair<point2d *, point2d *> > minmaxxy = merge(minmaxxy1, minmaxxy2);
  minmaxxy = merge(minmaxxy, minmaxxy3);
  minmaxxy = merge(minmaxxy, minmaxxy4);

  point2d l = *minmaxxy.first.first;
  point2d r = *minmaxxy.first.second;

  point2d b = *minmaxxy.second.first;
  point2d t = *minmaxxy.second.second;

  intT n1 = _Cilk_spawn wrapped_filter_new(P, offset_helper(Ptmp, 0),  n, aboveLineP(P, l, t));
  intT n2 = _Cilk_spawn wrapped_filter_new(P, offset_helper(Ptmp, num_pages), n, aboveLineP(P, t, r));
  intT n3 = _Cilk_spawn wrapped_filter_new(P, offset_helper(Ptmp, num_pages*2), n, aboveLineP(P, r, b));
  intT n4 = wrapped_filter_new(P, offset_helper(Ptmp, num_pages*3), n, aboveLineP(P, b, l));
  _Cilk_sync;
  intT m1; intT m2; intT m3; intT m4;
  m1 = _Cilk_spawn quickHullP(Ptmp, P, n1, l, t, 5);
  m2 = _Cilk_spawn quickHullP(offset_helper(Ptmp, num_pages), offset_helper(P, num_pages), n2, t, r, 5);
  m3 = _Cilk_spawn quickHullP(offset_helper(Ptmp, num_pages*2), offset_helper(P, num_pages*2), n3, r, b, 5);
  m4 = quickHullP(offset_helper(Ptmp, num_pages*3), offset_helper(P, num_pages*3), n4, b, l, 5);
  _Cilk_sync;

  int offset = 0;
  if (l.x != t.x || l.y != t.y){
    offset++;
  }
  _Cilk_for (intT i=0; i < m1; i++) P[i+offset] = Ptmp[i];
  if (t.x != r.x || t.y != r.y){
    offset++;
  }
  _Cilk_for (intT i=0; i < m2; i++) P[i+m1+offset] = offset_helper(Ptmp, num_pages)[i];
  if (r.x != b.x || r.y != b.y){
    offset++;
  }
  _Cilk_for (intT i=0; i < m3; i++) P[i+m1+m2+offset] = offset_helper(Ptmp, num_pages*2)[i];
  if (b.x != l.x || b.y != l.y){
    offset++;
  }
  _Cilk_for (intT i=0; i < m4; i++) P[i+m1+m2+m3+offset] = offset_helper(Ptmp, num_pages*3)[i];

  int offset2 = 0;

  P[0] = l;
  offset2 += m1;
  if(l.x != t.x || l.y != t.y){
    offset2++;
    P[offset2] = t;
  }

  offset2 += m2;
  if(t.x != r.x || t.y != r.y){
    offset2++;
    P[offset2] = r;
  }

  offset2 += m3;
  if(t.x != r.x || t.y != r.y){
    offset2++;
    P[offset2] = b;
  }
  return _seq<point2d>(P, offset2+1);
}

// CHECK-LABEL: @_Z5hullPP8_point2dIdElS1_(

// CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED]]:
// CHECK: invoke void @_Z14find_minmax_xyP8_point2dIdEl(%struct.pair* sret %minmaxxy1,
// CHECK-NEXT: to label %[[AFTERINVOKE:.+]] unwind
// CHECK: [[AFTERINVOKE]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]

// CHECK: detach within %[[SYNCREG]], label %[[DETACHED2:.+]], label %[[CONTINUE2:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED2]]:
// CHECK: invoke void @_Z14find_minmax_xyP8_point2dIdEl(%struct.pair* sret %minmaxxy2,
// CHECK-NEXT: to label %[[AFTERINVOKE2:.+]] unwind
// CHECK: [[AFTERINVOKE2]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE2]]

// CHECK: detach within %[[SYNCREG]], label %[[DETACHED3:.+]], label %[[CONTINUE3:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED3]]:
// CHECK: invoke void @_Z14find_minmax_xyP8_point2dIdEl(%struct.pair* sret %minmaxxy3,
// CHECK-NEXT: to label %[[AFTERINVOKE3:.+]] unwind
// CHECK: [[AFTERINVOKE3]]:
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE3]]

// CHECK: detach within %[[SYNCREG]], label %[[DETACHED4:.+]], label %[[CONTINUE4:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED4]]:
// CHECK: %[[RET4:.+]] = invoke i64 @_Z18wrapped_filter_newI8_point2dIdEl10aboveLinePET0_PT_S5_S3_T1_(
// CHECK-NEXT: to label %[[AFTERINVOKE4:.+]] unwind
// CHECK: [[AFTERINVOKE4]]:
// CHECK-NEXT: store i64 %[[RET4]]
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE4]]

// CHECK: detach within %[[SYNCREG]], label %[[DETACHED5:.+]], label %[[CONTINUE5:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED5]]:
// CHECK: %[[RET5:.+]] = invoke i64 @_Z18wrapped_filter_newI8_point2dIdEl10aboveLinePET0_PT_S5_S3_T1_(
// CHECK-NEXT: to label %[[AFTERINVOKE5:.+]] unwind
// CHECK: [[AFTERINVOKE5]]:
// CHECK-NEXT: store i64 %[[RET5]]
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE5]]

// CHECK: detach within %[[SYNCREG]], label %[[DETACHED6:.+]], label %[[CONTINUE6:.+]] unwind label %{{.+}}
// CHECK: [[DETACHED6]]:
// CHECK: %[[RET6:.+]] = invoke i64 @_Z18wrapped_filter_newI8_point2dIdEl10aboveLinePET0_PT_S5_S3_T1_(
// CHECK-NEXT: to label %[[AFTERINVOKE6:.+]] unwind
// CHECK: [[AFTERINVOKE6]]:
// CHECK-NEXT: store i64 %[[RET6]]
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE6]]
