struct rdar8650865 {
  struct rdar8650865 *first;
  int x;
};

int test_rdar8650865(struct rdar8650865 *s) {
  return ((((((s->first)->first)
    ->first)
    ->first)
    ->first)
    ->first)
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->first
    ->x;
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: 1:8: StructDecl=rdar8650865:1:8 (Definition) Extent=[1:1 - 4:2]
// CHECK: 2:23: FieldDecl=first:2:23 (Definition) Extent=[2:3 - 2:28]
// CHECK: 2:10: TypeRef=struct rdar8650865:1:8 Extent=[2:10 - 2:21]
// CHECK: 3:7: FieldDecl=x:3:7 (Definition) Extent=[3:3 - 3:8]
// CHECK: 6:5: FunctionDecl=test_rdar8650865:6:5 (Definition) Extent=[6:1 - 124:2]
// CHECK: 6:42: ParmDecl=s:6:42 (Definition) Extent=[6:22 - 6:43]
// CHECK: 6:29: TypeRef=struct rdar8650865:1:8 Extent=[6:29 - 6:40]
// CHECK: 6:45: CompoundStmt= Extent=[6:45 - 124:2]
// CHECK: 7:3: ReturnStmt= Extent=[7:3 - 123:8]
// CHECK: 123:7: MemberRefExpr=x:3:7 SingleRefName=[123:7 - 123:8] RefName=[123:7 - 123:8] Extent=[7:10 - 123:8]
// CHECK: 122:7: MemberRefExpr=first:2:23 SingleRefName=[122:7 - 122:12] RefName=[122:7 - 122:12] Extent=[7:10 - 122:12]
// CHECK: 121:7: MemberRefExpr=first:2:23 SingleRefName=[121:7 - 121:12] RefName=[121:7 - 121:12] Extent=[7:10 - 121:12]
// CHECK: 120:7: MemberRefExpr=first:2:23 SingleRefName=[120:7 - 120:12] RefName=[120:7 - 120:12] Extent=[7:10 - 120:12]
// CHECK: 119:7: MemberRefExpr=first:2:23 SingleRefName=[119:7 - 119:12] RefName=[119:7 - 119:12] Extent=[7:10 - 119:12]
// CHECK: 118:7: MemberRefExpr=first:2:23 SingleRefName=[118:7 - 118:12] RefName=[118:7 - 118:12] Extent=[7:10 - 118:12]
// CHECK: 117:7: MemberRefExpr=first:2:23 SingleRefName=[117:7 - 117:12] RefName=[117:7 - 117:12] Extent=[7:10 - 117:12]
// CHECK: 116:7: MemberRefExpr=first:2:23 SingleRefName=[116:7 - 116:12] RefName=[116:7 - 116:12] Extent=[7:10 - 116:12]
// CHECK: 115:7: MemberRefExpr=first:2:23 SingleRefName=[115:7 - 115:12] RefName=[115:7 - 115:12] Extent=[7:10 - 115:12]
// CHECK: 114:7: MemberRefExpr=first:2:23 SingleRefName=[114:7 - 114:12] RefName=[114:7 - 114:12] Extent=[7:10 - 114:12]
// CHECK: 113:7: MemberRefExpr=first:2:23 SingleRefName=[113:7 - 113:12] RefName=[113:7 - 113:12] Extent=[7:10 - 113:12]
// CHECK: 112:7: MemberRefExpr=first:2:23 SingleRefName=[112:7 - 112:12] RefName=[112:7 - 112:12] Extent=[7:10 - 112:12]
// CHECK: 111:7: MemberRefExpr=first:2:23 SingleRefName=[111:7 - 111:12] RefName=[111:7 - 111:12] Extent=[7:10 - 111:12]
// CHECK: 110:7: MemberRefExpr=first:2:23 SingleRefName=[110:7 - 110:12] RefName=[110:7 - 110:12] Extent=[7:10 - 110:12]
// CHECK: 109:7: MemberRefExpr=first:2:23 SingleRefName=[109:7 - 109:12] RefName=[109:7 - 109:12] Extent=[7:10 - 109:12]
// CHECK: 108:7: MemberRefExpr=first:2:23 SingleRefName=[108:7 - 108:12] RefName=[108:7 - 108:12] Extent=[7:10 - 108:12]
// CHECK: 107:7: MemberRefExpr=first:2:23 SingleRefName=[107:7 - 107:12] RefName=[107:7 - 107:12] Extent=[7:10 - 107:12]
// CHECK: 106:7: MemberRefExpr=first:2:23 SingleRefName=[106:7 - 106:12] RefName=[106:7 - 106:12] Extent=[7:10 - 106:12]
// CHECK: 105:7: MemberRefExpr=first:2:23 SingleRefName=[105:7 - 105:12] RefName=[105:7 - 105:12] Extent=[7:10 - 105:12]
// CHECK: 104:7: MemberRefExpr=first:2:23 SingleRefName=[104:7 - 104:12] RefName=[104:7 - 104:12] Extent=[7:10 - 104:12]
// CHECK: 103:7: MemberRefExpr=first:2:23 SingleRefName=[103:7 - 103:12] RefName=[103:7 - 103:12] Extent=[7:10 - 103:12]
// CHECK: 102:7: MemberRefExpr=first:2:23 SingleRefName=[102:7 - 102:12] RefName=[102:7 - 102:12] Extent=[7:10 - 102:12]
// CHECK: 101:7: MemberRefExpr=first:2:23 SingleRefName=[101:7 - 101:12] RefName=[101:7 - 101:12] Extent=[7:10 - 101:12]
// CHECK: 100:7: MemberRefExpr=first:2:23 SingleRefName=[100:7 - 100:12] RefName=[100:7 - 100:12] Extent=[7:10 - 100:12]
// CHECK: 99:7: MemberRefExpr=first:2:23 SingleRefName=[99:7 - 99:12] RefName=[99:7 - 99:12] Extent=[7:10 - 99:12]
// CHECK: 98:7: MemberRefExpr=first:2:23 SingleRefName=[98:7 - 98:12] RefName=[98:7 - 98:12] Extent=[7:10 - 98:12]
// CHECK: 97:7: MemberRefExpr=first:2:23 SingleRefName=[97:7 - 97:12] RefName=[97:7 - 97:12] Extent=[7:10 - 97:12]
// CHECK: 96:7: MemberRefExpr=first:2:23 SingleRefName=[96:7 - 96:12] RefName=[96:7 - 96:12] Extent=[7:10 - 96:12]
// CHECK: 95:7: MemberRefExpr=first:2:23 SingleRefName=[95:7 - 95:12] RefName=[95:7 - 95:12] Extent=[7:10 - 95:12]
// CHECK: 94:7: MemberRefExpr=first:2:23 SingleRefName=[94:7 - 94:12] RefName=[94:7 - 94:12] Extent=[7:10 - 94:12]
// CHECK: 93:7: MemberRefExpr=first:2:23 SingleRefName=[93:7 - 93:12] RefName=[93:7 - 93:12] Extent=[7:10 - 93:12]
// CHECK: 92:7: MemberRefExpr=first:2:23 SingleRefName=[92:7 - 92:12] RefName=[92:7 - 92:12] Extent=[7:10 - 92:12]
// CHECK: 91:7: MemberRefExpr=first:2:23 SingleRefName=[91:7 - 91:12] RefName=[91:7 - 91:12] Extent=[7:10 - 91:12]
// CHECK: 90:7: MemberRefExpr=first:2:23 SingleRefName=[90:7 - 90:12] RefName=[90:7 - 90:12] Extent=[7:10 - 90:12]
// CHECK: 89:7: MemberRefExpr=first:2:23 SingleRefName=[89:7 - 89:12] RefName=[89:7 - 89:12] Extent=[7:10 - 89:12]
// CHECK: 88:7: MemberRefExpr=first:2:23 SingleRefName=[88:7 - 88:12] RefName=[88:7 - 88:12] Extent=[7:10 - 88:12]
// CHECK: 87:7: MemberRefExpr=first:2:23 SingleRefName=[87:7 - 87:12] RefName=[87:7 - 87:12] Extent=[7:10 - 87:12]
// CHECK: 86:7: MemberRefExpr=first:2:23 SingleRefName=[86:7 - 86:12] RefName=[86:7 - 86:12] Extent=[7:10 - 86:12]
// CHECK: 85:7: MemberRefExpr=first:2:23 SingleRefName=[85:7 - 85:12] RefName=[85:7 - 85:12] Extent=[7:10 - 85:12]
// CHECK: 84:7: MemberRefExpr=first:2:23 SingleRefName=[84:7 - 84:12] RefName=[84:7 - 84:12] Extent=[7:10 - 84:12]
// CHECK: 83:7: MemberRefExpr=first:2:23 SingleRefName=[83:7 - 83:12] RefName=[83:7 - 83:12] Extent=[7:10 - 83:12]
// CHECK: 82:7: MemberRefExpr=first:2:23 SingleRefName=[82:7 - 82:12] RefName=[82:7 - 82:12] Extent=[7:10 - 82:12]
// CHECK: 81:7: MemberRefExpr=first:2:23 SingleRefName=[81:7 - 81:12] RefName=[81:7 - 81:12] Extent=[7:10 - 81:12]
// CHECK: 80:7: MemberRefExpr=first:2:23 SingleRefName=[80:7 - 80:12] RefName=[80:7 - 80:12] Extent=[7:10 - 80:12]
// CHECK: 79:7: MemberRefExpr=first:2:23 SingleRefName=[79:7 - 79:12] RefName=[79:7 - 79:12] Extent=[7:10 - 79:12]
// CHECK: 78:7: MemberRefExpr=first:2:23 SingleRefName=[78:7 - 78:12] RefName=[78:7 - 78:12] Extent=[7:10 - 78:12]
// CHECK: 77:7: MemberRefExpr=first:2:23 SingleRefName=[77:7 - 77:12] RefName=[77:7 - 77:12] Extent=[7:10 - 77:12]
// CHECK: 76:7: MemberRefExpr=first:2:23 SingleRefName=[76:7 - 76:12] RefName=[76:7 - 76:12] Extent=[7:10 - 76:12]
// CHECK: 75:7: MemberRefExpr=first:2:23 SingleRefName=[75:7 - 75:12] RefName=[75:7 - 75:12] Extent=[7:10 - 75:12]
// CHECK: 74:7: MemberRefExpr=first:2:23 SingleRefName=[74:7 - 74:12] RefName=[74:7 - 74:12] Extent=[7:10 - 74:12]
// CHECK: 73:7: MemberRefExpr=first:2:23 SingleRefName=[73:7 - 73:12] RefName=[73:7 - 73:12] Extent=[7:10 - 73:12]
// CHECK: 72:7: MemberRefExpr=first:2:23 SingleRefName=[72:7 - 72:12] RefName=[72:7 - 72:12] Extent=[7:10 - 72:12]
// CHECK: 71:7: MemberRefExpr=first:2:23 SingleRefName=[71:7 - 71:12] RefName=[71:7 - 71:12] Extent=[7:10 - 71:12]
// CHECK: 70:7: MemberRefExpr=first:2:23 SingleRefName=[70:7 - 70:12] RefName=[70:7 - 70:12] Extent=[7:10 - 70:12]
// CHECK: 69:7: MemberRefExpr=first:2:23 SingleRefName=[69:7 - 69:12] RefName=[69:7 - 69:12] Extent=[7:10 - 69:12]
// CHECK: 68:7: MemberRefExpr=first:2:23 SingleRefName=[68:7 - 68:12] RefName=[68:7 - 68:12] Extent=[7:10 - 68:12]
// CHECK: 67:7: MemberRefExpr=first:2:23 SingleRefName=[67:7 - 67:12] RefName=[67:7 - 67:12] Extent=[7:10 - 67:12]
// CHECK: 66:7: MemberRefExpr=first:2:23 SingleRefName=[66:7 - 66:12] RefName=[66:7 - 66:12] Extent=[7:10 - 66:12]
// CHECK: 65:7: MemberRefExpr=first:2:23 SingleRefName=[65:7 - 65:12] RefName=[65:7 - 65:12] Extent=[7:10 - 65:12]
// CHECK: 64:7: MemberRefExpr=first:2:23 SingleRefName=[64:7 - 64:12] RefName=[64:7 - 64:12] Extent=[7:10 - 64:12]
// CHECK: 63:7: MemberRefExpr=first:2:23 SingleRefName=[63:7 - 63:12] RefName=[63:7 - 63:12] Extent=[7:10 - 63:12]
// CHECK: 62:7: MemberRefExpr=first:2:23 SingleRefName=[62:7 - 62:12] RefName=[62:7 - 62:12] Extent=[7:10 - 62:12]
// CHECK: 61:7: MemberRefExpr=first:2:23 SingleRefName=[61:7 - 61:12] RefName=[61:7 - 61:12] Extent=[7:10 - 61:12]
// CHECK: 60:7: MemberRefExpr=first:2:23 SingleRefName=[60:7 - 60:12] RefName=[60:7 - 60:12] Extent=[7:10 - 60:12]
// CHECK: 59:7: MemberRefExpr=first:2:23 SingleRefName=[59:7 - 59:12] RefName=[59:7 - 59:12] Extent=[7:10 - 59:12]
// CHECK: 58:7: MemberRefExpr=first:2:23 SingleRefName=[58:7 - 58:12] RefName=[58:7 - 58:12] Extent=[7:10 - 58:12]
// CHECK: 57:7: MemberRefExpr=first:2:23 SingleRefName=[57:7 - 57:12] RefName=[57:7 - 57:12] Extent=[7:10 - 57:12]
// CHECK: 56:7: MemberRefExpr=first:2:23 SingleRefName=[56:7 - 56:12] RefName=[56:7 - 56:12] Extent=[7:10 - 56:12]
// CHECK: 55:7: MemberRefExpr=first:2:23 SingleRefName=[55:7 - 55:12] RefName=[55:7 - 55:12] Extent=[7:10 - 55:12]
// CHECK: 54:7: MemberRefExpr=first:2:23 SingleRefName=[54:7 - 54:12] RefName=[54:7 - 54:12] Extent=[7:10 - 54:12]
// CHECK: 53:7: MemberRefExpr=first:2:23 SingleRefName=[53:7 - 53:12] RefName=[53:7 - 53:12] Extent=[7:10 - 53:12]
// CHECK: 52:7: MemberRefExpr=first:2:23 SingleRefName=[52:7 - 52:12] RefName=[52:7 - 52:12] Extent=[7:10 - 52:12]
// CHECK: 51:7: MemberRefExpr=first:2:23 SingleRefName=[51:7 - 51:12] RefName=[51:7 - 51:12] Extent=[7:10 - 51:12]
// CHECK: 50:7: MemberRefExpr=first:2:23 SingleRefName=[50:7 - 50:12] RefName=[50:7 - 50:12] Extent=[7:10 - 50:12]
// CHECK: 49:7: MemberRefExpr=first:2:23 SingleRefName=[49:7 - 49:12] RefName=[49:7 - 49:12] Extent=[7:10 - 49:12]
// CHECK: 48:7: MemberRefExpr=first:2:23 SingleRefName=[48:7 - 48:12] RefName=[48:7 - 48:12] Extent=[7:10 - 48:12]
// CHECK: 47:7: MemberRefExpr=first:2:23 SingleRefName=[47:7 - 47:12] RefName=[47:7 - 47:12] Extent=[7:10 - 47:12]
// CHECK: 46:7: MemberRefExpr=first:2:23 SingleRefName=[46:7 - 46:12] RefName=[46:7 - 46:12] Extent=[7:10 - 46:12]
// CHECK: 45:7: MemberRefExpr=first:2:23 SingleRefName=[45:7 - 45:12] RefName=[45:7 - 45:12] Extent=[7:10 - 45:12]
// CHECK: 44:7: MemberRefExpr=first:2:23 SingleRefName=[44:7 - 44:12] RefName=[44:7 - 44:12] Extent=[7:10 - 44:12]
// CHECK: 43:7: MemberRefExpr=first:2:23 SingleRefName=[43:7 - 43:12] RefName=[43:7 - 43:12] Extent=[7:10 - 43:12]
// CHECK: 42:7: MemberRefExpr=first:2:23 SingleRefName=[42:7 - 42:12] RefName=[42:7 - 42:12] Extent=[7:10 - 42:12]
// CHECK: 41:7: MemberRefExpr=first:2:23 SingleRefName=[41:7 - 41:12] RefName=[41:7 - 41:12] Extent=[7:10 - 41:12]
// CHECK: 40:7: MemberRefExpr=first:2:23 SingleRefName=[40:7 - 40:12] RefName=[40:7 - 40:12] Extent=[7:10 - 40:12]
// CHECK: 39:7: MemberRefExpr=first:2:23 SingleRefName=[39:7 - 39:12] RefName=[39:7 - 39:12] Extent=[7:10 - 39:12]
// CHECK: 38:7: MemberRefExpr=first:2:23 SingleRefName=[38:7 - 38:12] RefName=[38:7 - 38:12] Extent=[7:10 - 38:12]
// CHECK: 37:7: MemberRefExpr=first:2:23 SingleRefName=[37:7 - 37:12] RefName=[37:7 - 37:12] Extent=[7:10 - 37:12]
// CHECK: 36:7: MemberRefExpr=first:2:23 SingleRefName=[36:7 - 36:12] RefName=[36:7 - 36:12] Extent=[7:10 - 36:12]
// CHECK: 35:7: MemberRefExpr=first:2:23 SingleRefName=[35:7 - 35:12] RefName=[35:7 - 35:12] Extent=[7:10 - 35:12]
// CHECK: 34:7: MemberRefExpr=first:2:23 SingleRefName=[34:7 - 34:12] RefName=[34:7 - 34:12] Extent=[7:10 - 34:12]
// CHECK: 33:7: MemberRefExpr=first:2:23 SingleRefName=[33:7 - 33:12] RefName=[33:7 - 33:12] Extent=[7:10 - 33:12]
// CHECK: 32:7: MemberRefExpr=first:2:23 SingleRefName=[32:7 - 32:12] RefName=[32:7 - 32:12] Extent=[7:10 - 32:12]
// CHECK: 31:7: MemberRefExpr=first:2:23 SingleRefName=[31:7 - 31:12] RefName=[31:7 - 31:12] Extent=[7:10 - 31:12]
// CHECK: 30:7: MemberRefExpr=first:2:23 SingleRefName=[30:7 - 30:12] RefName=[30:7 - 30:12] Extent=[7:10 - 30:12]
// CHECK: 29:7: MemberRefExpr=first:2:23 SingleRefName=[29:7 - 29:12] RefName=[29:7 - 29:12] Extent=[7:10 - 29:12]
// CHECK: 28:7: MemberRefExpr=first:2:23 SingleRefName=[28:7 - 28:12] RefName=[28:7 - 28:12] Extent=[7:10 - 28:12]
// CHECK: 27:7: MemberRefExpr=first:2:23 SingleRefName=[27:7 - 27:12] RefName=[27:7 - 27:12] Extent=[7:10 - 27:12]
// CHECK: 26:7: MemberRefExpr=first:2:23 SingleRefName=[26:7 - 26:12] RefName=[26:7 - 26:12] Extent=[7:10 - 26:12]
// CHECK: 25:7: MemberRefExpr=first:2:23 SingleRefName=[25:7 - 25:12] RefName=[25:7 - 25:12] Extent=[7:10 - 25:12]
// CHECK: 24:7: MemberRefExpr=first:2:23 SingleRefName=[24:7 - 24:12] RefName=[24:7 - 24:12] Extent=[7:10 - 24:12]
// CHECK: 23:7: MemberRefExpr=first:2:23 SingleRefName=[23:7 - 23:12] RefName=[23:7 - 23:12] Extent=[7:10 - 23:12]
// CHECK: 22:7: MemberRefExpr=first:2:23 SingleRefName=[22:7 - 22:12] RefName=[22:7 - 22:12] Extent=[7:10 - 22:12]
// CHECK: 21:7: MemberRefExpr=first:2:23 SingleRefName=[21:7 - 21:12] RefName=[21:7 - 21:12] Extent=[7:10 - 21:12]
// CHECK: 20:7: MemberRefExpr=first:2:23 SingleRefName=[20:7 - 20:12] RefName=[20:7 - 20:12] Extent=[7:10 - 20:12]
// CHECK: 19:7: MemberRefExpr=first:2:23 SingleRefName=[19:7 - 19:12] RefName=[19:7 - 19:12] Extent=[7:10 - 19:12]
// CHECK: 18:7: MemberRefExpr=first:2:23 SingleRefName=[18:7 - 18:12] RefName=[18:7 - 18:12] Extent=[7:10 - 18:12]
// CHECK: 17:7: MemberRefExpr=first:2:23 SingleRefName=[17:7 - 17:12] RefName=[17:7 - 17:12] Extent=[7:10 - 17:12]
// CHECK: 16:7: MemberRefExpr=first:2:23 SingleRefName=[16:7 - 16:12] RefName=[16:7 - 16:12] Extent=[7:10 - 16:12]
// CHECK: 15:7: MemberRefExpr=first:2:23 SingleRefName=[15:7 - 15:12] RefName=[15:7 - 15:12] Extent=[7:10 - 15:12]
// CHECK: 14:7: MemberRefExpr=first:2:23 SingleRefName=[14:7 - 14:12] RefName=[14:7 - 14:12] Extent=[7:10 - 14:12]
// CHECK: 13:7: MemberRefExpr=first:2:23 SingleRefName=[13:7 - 13:12] RefName=[13:7 - 13:12] Extent=[7:10 - 13:12]
// CHECK: 12:7: MemberRefExpr=first:2:23 SingleRefName=[12:7 - 12:12] RefName=[12:7 - 12:12] Extent=[7:10 - 12:12]
// CHECK: 11:7: MemberRefExpr=first:2:23 SingleRefName=[11:7 - 11:12] RefName=[11:7 - 11:12] Extent=[7:11 - 11:12]
// CHECK: 10:7: MemberRefExpr=first:2:23 SingleRefName=[10:7 - 10:12] RefName=[10:7 - 10:12] Extent=[7:12 - 10:12]
// CHECK: 9:7: MemberRefExpr=first:2:23 SingleRefName=[9:7 - 9:12] RefName=[9:7 - 9:12] Extent=[7:13 - 9:12]
// CHECK: 8:7: MemberRefExpr=first:2:23 SingleRefName=[8:7 - 8:12] RefName=[8:7 - 8:12] Extent=[7:14 - 8:12]
// CHECK: 7:27: MemberRefExpr=first:2:23 SingleRefName=[7:27 - 7:32] RefName=[7:27 - 7:32] Extent=[7:15 - 7:32]
// CHECK: 7:19: MemberRefExpr=first:2:23 SingleRefName=[7:19 - 7:24] RefName=[7:19 - 7:24] Extent=[7:16 - 7:24]
// CHECK: 7:16: DeclRefExpr=s:6:42 Extent=[7:16 - 7:17]

// RUN: c-index-test -test-annotate-tokens=%s:1:1:124:1 %s | FileCheck -check-prefix=CHECK-tokens %s
// CHECK-tokens: Keyword: "struct" [1:1 - 1:7] StructDecl=rdar8650865:1:8 (Definition)
// CHECK-tokens: Identifier: "rdar8650865" [1:8 - 1:19] StructDecl=rdar8650865:1:8 (Definition)
// CHECK-tokens: Punctuation: "{" [1:20 - 1:21] StructDecl=rdar8650865:1:8 (Definition)
// CHECK-tokens: Keyword: "struct" [2:3 - 2:9] FieldDecl=first:2:23 (Definition)
// CHECK-tokens: Identifier: "rdar8650865" [2:10 - 2:21] TypeRef=struct rdar8650865:1:8
// CHECK-tokens: Punctuation: "*" [2:22 - 2:23] FieldDecl=first:2:23 (Definition)
// CHECK-tokens: Identifier: "first" [2:23 - 2:28] FieldDecl=first:2:23 (Definition)
// CHECK-tokens: Punctuation: ";" [2:28 - 2:29] StructDecl=rdar8650865:1:8 (Definition)
// CHECK-tokens: Keyword: "int" [3:3 - 3:6] FieldDecl=x:3:7 (Definition)
// CHECK-tokens: Identifier: "x" [3:7 - 3:8] FieldDecl=x:3:7 (Definition)
// CHECK-tokens: Punctuation: ";" [3:8 - 3:9] StructDecl=rdar8650865:1:8 (Definition)
// CHECK-tokens: Punctuation: "}" [4:1 - 4:2] StructDecl=rdar8650865:1:8 (Definition)
// CHECK-tokens: Punctuation: ";" [4:2 - 4:3]
// CHECK-tokens: Keyword: "int" [6:1 - 6:4] FunctionDecl=test_rdar8650865:6:5 (Definition)
// CHECK-tokens: Identifier: "test_rdar8650865" [6:5 - 6:21] FunctionDecl=test_rdar8650865:6:5 (Definition)
// CHECK-tokens: Punctuation: "(" [6:21 - 6:22] FunctionDecl=test_rdar8650865:6:5 (Definition)
// CHECK-tokens: Keyword: "struct" [6:22 - 6:28] ParmDecl=s:6:42 (Definition)
// CHECK-tokens: Identifier: "rdar8650865" [6:29 - 6:40] TypeRef=struct rdar8650865:1:8
// CHECK-tokens: Punctuation: "*" [6:41 - 6:42] ParmDecl=s:6:42 (Definition)
// CHECK-tokens: Identifier: "s" [6:42 - 6:43] ParmDecl=s:6:42 (Definition)
// CHECK-tokens: Punctuation: ")" [6:43 - 6:44] FunctionDecl=test_rdar8650865:6:5 (Definition)
// CHECK-tokens: Punctuation: "{" [6:45 - 6:46] CompoundStmt=
// CHECK-tokens: Keyword: "return" [7:3 - 7:9] ReturnStmt=
// CHECK-tokens: Punctuation: "(" [7:10 - 7:11] ParenExpr=
// CHECK-tokens: Punctuation: "(" [7:11 - 7:12] ParenExpr=
// CHECK-tokens: Punctuation: "(" [7:12 - 7:13] ParenExpr=
// CHECK-tokens: Punctuation: "(" [7:13 - 7:14] ParenExpr=
// CHECK-tokens: Punctuation: "(" [7:14 - 7:15] ParenExpr=
// CHECK-tokens: Punctuation: "(" [7:15 - 7:16] ParenExpr=
// CHECK-tokens: Identifier: "s" [7:16 - 7:17] DeclRefExpr=s:6:42
// CHECK-tokens: Punctuation: "->" [7:17 - 7:19] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [7:19 - 7:24] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: ")" [7:24 - 7:25] ParenExpr=
// CHECK-tokens: Punctuation: "->" [7:25 - 7:27] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [7:27 - 7:32] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: ")" [7:32 - 7:33] ParenExpr=
// CHECK-tokens: Punctuation: "->" [8:5 - 8:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [8:7 - 8:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: ")" [8:12 - 8:13] ParenExpr=
// CHECK-tokens: Punctuation: "->" [9:5 - 9:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [9:7 - 9:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: ")" [9:12 - 9:13] ParenExpr=
// CHECK-tokens: Punctuation: "->" [10:5 - 10:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [10:7 - 10:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: ")" [10:12 - 10:13] ParenExpr=
// CHECK-tokens: Punctuation: "->" [11:5 - 11:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [11:7 - 11:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: ")" [11:12 - 11:13] ParenExpr=
// CHECK-tokens: Punctuation: "->" [12:5 - 12:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [12:7 - 12:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [13:5 - 13:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [13:7 - 13:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [14:5 - 14:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [14:7 - 14:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [15:5 - 15:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [15:7 - 15:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [16:5 - 16:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [16:7 - 16:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [17:5 - 17:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [17:7 - 17:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [18:5 - 18:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [18:7 - 18:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [19:5 - 19:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [19:7 - 19:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [20:5 - 20:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [20:7 - 20:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [21:5 - 21:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [21:7 - 21:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [22:5 - 22:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [22:7 - 22:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [23:5 - 23:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [23:7 - 23:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [24:5 - 24:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [24:7 - 24:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [25:5 - 25:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [25:7 - 25:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [26:5 - 26:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [26:7 - 26:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [27:5 - 27:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [27:7 - 27:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [28:5 - 28:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [28:7 - 28:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [29:5 - 29:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [29:7 - 29:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [30:5 - 30:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [30:7 - 30:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [31:5 - 31:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [31:7 - 31:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [32:5 - 32:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [32:7 - 32:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [33:5 - 33:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [33:7 - 33:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [34:5 - 34:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [34:7 - 34:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [35:5 - 35:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [35:7 - 35:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [36:5 - 36:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [36:7 - 36:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [37:5 - 37:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [37:7 - 37:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [38:5 - 38:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [38:7 - 38:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [39:5 - 39:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [39:7 - 39:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [40:5 - 40:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [40:7 - 40:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [41:5 - 41:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [41:7 - 41:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [42:5 - 42:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [42:7 - 42:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [43:5 - 43:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [43:7 - 43:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [44:5 - 44:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [44:7 - 44:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [45:5 - 45:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [45:7 - 45:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [46:5 - 46:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [46:7 - 46:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [47:5 - 47:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [47:7 - 47:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [48:5 - 48:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [48:7 - 48:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [49:5 - 49:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [49:7 - 49:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [50:5 - 50:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [50:7 - 50:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [51:5 - 51:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [51:7 - 51:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [52:5 - 52:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [52:7 - 52:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [53:5 - 53:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [53:7 - 53:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [54:5 - 54:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [54:7 - 54:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [55:5 - 55:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [55:7 - 55:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [56:5 - 56:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [56:7 - 56:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [57:5 - 57:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [57:7 - 57:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [58:5 - 58:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [58:7 - 58:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [59:5 - 59:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [59:7 - 59:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [60:5 - 60:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [60:7 - 60:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [61:5 - 61:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [61:7 - 61:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [62:5 - 62:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [62:7 - 62:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [63:5 - 63:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [63:7 - 63:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [64:5 - 64:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [64:7 - 64:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [65:5 - 65:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [65:7 - 65:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [66:5 - 66:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [66:7 - 66:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [67:5 - 67:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [67:7 - 67:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [68:5 - 68:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [68:7 - 68:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [69:5 - 69:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [69:7 - 69:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [70:5 - 70:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [70:7 - 70:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [71:5 - 71:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [71:7 - 71:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [72:5 - 72:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [72:7 - 72:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [73:5 - 73:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [73:7 - 73:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [74:5 - 74:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [74:7 - 74:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [75:5 - 75:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [75:7 - 75:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [76:5 - 76:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [76:7 - 76:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [77:5 - 77:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [77:7 - 77:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [78:5 - 78:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [78:7 - 78:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [79:5 - 79:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [79:7 - 79:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [80:5 - 80:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [80:7 - 80:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [81:5 - 81:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [81:7 - 81:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [82:5 - 82:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [82:7 - 82:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [83:5 - 83:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [83:7 - 83:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [84:5 - 84:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [84:7 - 84:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [85:5 - 85:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [85:7 - 85:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [86:5 - 86:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [86:7 - 86:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [87:5 - 87:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [87:7 - 87:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [88:5 - 88:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [88:7 - 88:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [89:5 - 89:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [89:7 - 89:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [90:5 - 90:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [90:7 - 90:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [91:5 - 91:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [91:7 - 91:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [92:5 - 92:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [92:7 - 92:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [93:5 - 93:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [93:7 - 93:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [94:5 - 94:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [94:7 - 94:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [95:5 - 95:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [95:7 - 95:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [96:5 - 96:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [96:7 - 96:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [97:5 - 97:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [97:7 - 97:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [98:5 - 98:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [98:7 - 98:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [99:5 - 99:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [99:7 - 99:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [100:5 - 100:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [100:7 - 100:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [101:5 - 101:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [101:7 - 101:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [102:5 - 102:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [102:7 - 102:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [103:5 - 103:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [103:7 - 103:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [104:5 - 104:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [104:7 - 104:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [105:5 - 105:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [105:7 - 105:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [106:5 - 106:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [106:7 - 106:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [107:5 - 107:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [107:7 - 107:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [108:5 - 108:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [108:7 - 108:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [109:5 - 109:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [109:7 - 109:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [110:5 - 110:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [110:7 - 110:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [111:5 - 111:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [111:7 - 111:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [112:5 - 112:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [112:7 - 112:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [113:5 - 113:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [113:7 - 113:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [114:5 - 114:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [114:7 - 114:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [115:5 - 115:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [115:7 - 115:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [116:5 - 116:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [116:7 - 116:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [117:5 - 117:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [117:7 - 117:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [118:5 - 118:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [118:7 - 118:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [119:5 - 119:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [119:7 - 119:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [120:5 - 120:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [120:7 - 120:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [121:5 - 121:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [121:7 - 121:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [122:5 - 122:7] MemberRefExpr=first:2:23
// CHECK-tokens: Identifier: "first" [122:7 - 122:12] MemberRefExpr=first:2:23
// CHECK-tokens: Punctuation: "->" [123:5 - 123:7] MemberRefExpr=x:3:7
// CHECK-tokens: Identifier: "x" [123:7 - 123:8] MemberRefExpr=x:3:7
// CHECK-tokens: Punctuation: ";" [123:8 - 123:9] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [124:1 - 124:2] CompoundStmt=


