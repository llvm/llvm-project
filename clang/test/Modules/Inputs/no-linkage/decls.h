namespace RealNS { int UsingDecl; }
namespace NS = RealNS;
typedef int Typedef;
using AliasDecl = int;
using RealNS::UsingDecl;
struct Struct {};
extern int Variable;
namespace AnotherNS {}
enum X { Enumerator };
void Overloads();
void Overloads(int);
