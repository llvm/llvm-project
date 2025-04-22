/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Attribute name matcher                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Attr.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static AttributeCommonInfo::Kind getAttrKind(StringRef Name, AttributeCommonInfo::Syntax Syntax) {
  if (AttributeCommonInfo::AS_GNU == Syntax) {
  switch (Name.size()) {
  default: break;
  case 3:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ar", 2) != 0)
        break;
      return AttributeCommonInfo::AT_MipsLongCall;	 // "far"
    case 'h':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ot", 2) != 0)
        break;
      return AttributeCommonInfo::AT_Hot;	 // "hot"
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "cs", 2) != 0)
        break;
      return AttributeCommonInfo::AT_Pcs;	 // "pcs"
    }
    break;
  case 4:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "old", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Cold;	 // "cold"
    case 'h':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ost", 3) != 0)
        break;
      return AttributeCommonInfo::AT_CUDAHost;	 // "host"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eaf", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Leaf;	 // "leaf"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ode", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Mode;	 // "mode"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ear", 3) != 0)
        break;
      return AttributeCommonInfo::AT_MipsShortCall;	 // "near"
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ure", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Pure;	 // "pure"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "sed", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Used;	 // "used"
    case 'w':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eak", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Weak;	 // "weak"
    }
    break;
  case 5:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lias", 4) != 0)
        break;
      return AttributeCommonInfo::AT_Alias;	 // "alias"
    case 'c':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ecl", 3) != 0)
          break;
        return AttributeCommonInfo::AT_CDecl;	 // "cdecl"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "nst", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Const;	 // "const"
      }
      break;
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rror", 4) != 0)
        break;
      return AttributeCommonInfo::AT_Error;	 // "error"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uard", 4) != 0)
        break;
      return AttributeCommonInfo::AT_CFGuard;	 // "guard"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "func", 4) != 0)
        break;
      return AttributeCommonInfo::AT_IFunc;	 // "ifunc"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "odel", 4) != 0)
        break;
      return AttributeCommonInfo::AT_CodeModel;	 // "model"
    case 'n':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ked", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Naked;	 // "naked"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "svm", 3) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLNoSVM;	 // "nosvm"
      }
      break;
    }
    break;
  case 6:	 // 17 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sume", 4) != 0)
          break;
        return AttributeCommonInfo::AT_CXXAssume;	 // "assume"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "omic", 4) != 0)
          break;
        return AttributeCommonInfo::AT_Atomic;	 // "atomic"
      }
      break;
    case 'b':	 // 1 string to match.
      if (memcmp(Name.data()+1, "locks", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Blocks;	 // "blocks"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ommon", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Common;	 // "common"
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "evice", 5) != 0)
        break;
      return AttributeCommonInfo::AT_CUDADevice;	 // "device"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ormat", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Format;	 // "format"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lobal", 5) != 0)
        break;
      return AttributeCommonInfo::AT_CUDAGlobal;	 // "global"
    case 'm':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "lloc", 4) != 0)
          break;
        return AttributeCommonInfo::AT_Restrict;	 // "malloc"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ps16", 4) != 0)
          break;
        return AttributeCommonInfo::AT_Mips16;	 // "mips16"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_abi", 4) != 0)
          break;
        return AttributeCommonInfo::AT_MSABI;	 // "ms_abi"
      }
      break;
    case 'p':	 // 2 strings to match.
      if (Name[1] != 'a')
        break;
      switch (Name[2]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+3, "ked", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Packed;	 // "packed"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+3, "cal", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Pascal;	 // "pascal"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "etain", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Retain;	 // "retain"
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ared", 4) != 0)
          break;
        return AttributeCommonInfo::AT_CUDAShared;	 // "shared"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "gnal", 4) != 0)
          break;
        return AttributeCommonInfo::AT_AVRSignal;	 // "signal"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "arget", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Target;	 // "target"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nused", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Unused;	 // "unused"
    }
    break;
  case 7:	 // 22 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_const", 6) != 0)
        break;
      return AttributeCommonInfo::AT_Const;	 // "__const"
    case 'a':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+2, "i_tag", 5) != 0)
          break;
        return AttributeCommonInfo::AT_AbiTag;	 // "abi_tag"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+2, "igned", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Aligned;	 // "aligned"
      }
      break;
    case 'b':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ounded", 6) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "bounded"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "leanup", 6) != 0)
        break;
      return AttributeCommonInfo::AT_Cleanup;	 // "cleanup"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "latten", 6) != 0)
        break;
      return AttributeCommonInfo::AT_Flatten;	 // "flatten"
    case 'm':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "naged", 5) != 0)
          break;
        return AttributeCommonInfo::AT_HIPManaged;	 // "managed"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "nsize", 5) != 0)
          break;
        return AttributeCommonInfo::AT_MinSize;	 // "minsize"
      }
      break;
    case 'n':	 // 6 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 5 strings to match.
        switch (Name[2]) {
        default: break;
        case 'd':	 // 2 strings to match.
          if (Name[3] != 'e')
            break;
          switch (Name[4]) {
          default: break;
          case 'b':	 // 1 string to match.
            if (memcmp(Name.data()+5, "ug", 2) != 0)
              break;
            return AttributeCommonInfo::AT_NoDebug;	 // "nodebug"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+5, "ef", 2) != 0)
              break;
            return AttributeCommonInfo::AT_NoDeref;	 // "noderef"
          }
          break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+3, "erge", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoMerge;	 // "nomerge"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+3, "null", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NonNull;	 // "nonnull"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+3, "hrow", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoThrow;	 // "nothrow"
        }
        break;
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_weak", 5) != 0)
          break;
        return AttributeCommonInfo::IgnoredAttribute;	 // "nv_weak"
      }
      break;
    case 'o':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+2, "jc_gc", 5) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCGC;	 // "objc_gc"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "tnone", 5) != 0)
          break;
        return AttributeCommonInfo::AT_OptimizeNone;	 // "optnone"
      }
      break;
    case 'r':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "eg", 2) != 0)
        break;
      switch (Name[3]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+4, "all", 3) != 0)
          break;
        return AttributeCommonInfo::AT_RegCall;	 // "regcall"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+4, "arm", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Regparm;	 // "regparm"
      }
      break;
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ction", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Section;	 // "section"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "dcall", 5) != 0)
          break;
        return AttributeCommonInfo::AT_StdCall;	 // "stdcall"
      }
      break;
    case 'w':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rning", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Error;	 // "warning"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "akref", 5) != 0)
          break;
        return AttributeCommonInfo::AT_WeakRef;	 // "weakref"
      }
      break;
    }
    break;
  case 8:	 // 21 strings to match.
    switch (Name[0]) {
    default: break;
    case 'N':	 // 1 string to match.
      if (memcmp(Name.data()+1, "SObject", 7) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCNSObject;	 // "NSObject"
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nnotate", 7) != 0)
        break;
      return AttributeCommonInfo::AT_Annotate;	 // "annotate"
    case 'b':	 // 1 string to match.
      if (memcmp(Name.data()+1, "locking", 7) != 0)
        break;
      return AttributeCommonInfo::AT_Blocking;	 // "blocking"
    case 'c':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "llback", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Callback;	 // "callback"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "nstant", 6) != 0)
          break;
        return AttributeCommonInfo::AT_CUDAConstant;	 // "constant"
      }
      break;
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "astcall", 7) != 0)
        break;
      return AttributeCommonInfo::AT_FastCall;	 // "fastcall"
    case 'i':	 // 2 strings to match.
      if (Name[1] != 'b')
        break;
      switch (Name[2]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+3, "ction", 5) != 0)
          break;
        return AttributeCommonInfo::AT_IBAction;	 // "ibaction"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+3, "utlet", 5) != 0)
          break;
        return AttributeCommonInfo::AT_IBOutlet;	 // "iboutlet"
      }
      break;
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ockable", 7) != 0)
        break;
      return AttributeCommonInfo::AT_Lockable;	 // "lockable"
    case 'm':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case '6':	 // 1 string to match.
        if (memcmp(Name.data()+2, "8k_rtd", 6) != 0)
          break;
        return AttributeCommonInfo::AT_M68kRTD;	 // "m68k_rtd"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sttail", 6) != 0)
          break;
        return AttributeCommonInfo::AT_MustTail;	 // "musttail"
      }
      break;
    case 'n':	 // 5 strings to match.
      if (Name[1] != 'o')
        break;
      switch (Name[2]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+3, "ommon", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoCommon;	 // "nocommon"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+3, "scape", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoEscape;	 // "noescape"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+3, "nline", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoInline;	 // "noinline"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+3, "ips16", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoMips16;	 // "nomips16"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+3, "eturn", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoReturn;	 // "noreturn"
      }
      break;
    case 's':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ntinel", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Sentinel;	 // "sentinel"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "zed_by", 6) != 0)
          break;
        return AttributeCommonInfo::AT_SizedBy;	 // "sized_by"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ppress", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Suppress;	 // "suppress"
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sv_abi", 6) != 0)
          break;
        return AttributeCommonInfo::AT_SysVABI;	 // "sysv_abi"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "hiscall", 7) != 0)
        break;
      return AttributeCommonInfo::AT_ThisCall;	 // "thiscall"
    }
    break;
  case 9:	 // 14 strings to match.
    switch (Name[0]) {
    default: break;
    case 'd':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "ll", 2) != 0)
        break;
      switch (Name[3]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+4, "xport", 5) != 0)
          break;
        return AttributeCommonInfo::AT_DLLExport;	 // "dllexport"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+4, "mport", 5) != 0)
          break;
        return AttributeCommonInfo::AT_DLLImport;	 // "dllimport"
      }
      break;
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nable_if", 8) != 0)
        break;
      return AttributeCommonInfo::AT_EnableIf;	 // "enable_if"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lag_enum", 8) != 0)
        break;
      return AttributeCommonInfo::AT_FlagEnum;	 // "flag_enum"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nterrupt", 8) != 0)
        break;
      return AttributeCommonInfo::AT_Interrupt;	 // "interrupt"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ong_call", 8) != 0)
        break;
      return AttributeCommonInfo::AT_MipsLongCall;	 // "long_call"
    case 'm':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "y_alias", 7) != 0)
          break;
        return AttributeCommonInfo::AT_MayAlias;	 // "may_alias"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cromips", 7) != 0)
          break;
        return AttributeCommonInfo::AT_MicroMips;	 // "micromips"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_struct", 7) != 0)
          break;
        return AttributeCommonInfo::AT_MSStruct;	 // "ms_struct"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ouwtable", 8) != 0)
        break;
      return AttributeCommonInfo::AT_NoUwtable;	 // "nouwtable"
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "lectany", 7) != 0)
          break;
        return AttributeCommonInfo::AT_SelectAny;	 // "selectany"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+2, "iftcall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftCall;	 // "swiftcall"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ls_model", 8) != 0)
        break;
      return AttributeCommonInfo::AT_TLSModel;	 // "tls_model"
    case 'v':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ecreturn", 8) != 0)
        break;
      return AttributeCommonInfo::AT_VecReturn;	 // "vecreturn"
    }
    break;
  case 10:	 // 22 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'l':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "loc", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case '_':	 // 1 string to match.
          if (memcmp(Name.data()+6, "size", 4) != 0)
            break;
          return AttributeCommonInfo::AT_AllocSize;	 // "alloc_size"
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+6, "ting", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Allocating;	 // "allocating"
        }
        break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "tificial", 8) != 0)
          break;
        return AttributeCommonInfo::AT_Artificial;	 // "artificial"
      }
      break;
    case 'c':	 // 5 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "pability", 8) != 0)
          break;
        return AttributeCommonInfo::AT_Capability;	 // "capability"
      case 'o':	 // 4 strings to match.
        switch (Name[2]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+3, "e_align", 7) != 0)
            break;
          return AttributeCommonInfo::AT_CodeAlign;	 // "code_align"
        case 'n':	 // 2 strings to match.
          switch (Name[3]) {
          default: break;
          case 's':	 // 1 string to match.
            if (memcmp(Name.data()+4, "umable", 6) != 0)
              break;
            return AttributeCommonInfo::AT_Consumable;	 // "consumable"
          case 'v':	 // 1 string to match.
            if (memcmp(Name.data()+4, "ergent", 6) != 0)
              break;
            return AttributeCommonInfo::AT_Convergent;	 // "convergent"
          }
          break;
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+3, "nted_by", 7) != 0)
            break;
          return AttributeCommonInfo::AT_CountedBy;	 // "counted_by"
        }
        break;
      }
      break;
    case 'd':	 // 2 strings to match.
      if (Name[1] != 'e')
        break;
      switch (Name[2]) {
      default: break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+3, "recated", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Deprecated;	 // "deprecated"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+3, "tructor", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Destructor;	 // "destructor"
      }
      break;
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ormat_arg", 9) != 0)
        break;
      return AttributeCommonInfo::AT_FormatArg;	 // "format_arg"
    case 'g':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+2, "u_inline", 8) != 0)
          break;
        return AttributeCommonInfo::AT_GNUInline;	 // "gnu_inline"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+2, "arded_by", 8) != 0)
          break;
        return AttributeCommonInfo::AT_GuardedBy;	 // "guarded_by"
      }
      break;
    case 'n':	 // 3 strings to match.
      if (Name[1] != 'o')
        break;
      switch (Name[2]) {
      default: break;
      case '_':	 // 2 strings to match.
        switch (Name[3]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+4, "uiltin", 6) != 0)
            break;
          return AttributeCommonInfo::AT_NoBuiltin;	 // "no_builtin"
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+4, "estroy", 6) != 0)
            break;
          return AttributeCommonInfo::AT_NoDestroy;	 // "no_destroy"
        }
        break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+3, "f_check", 7) != 0)
          break;
        return AttributeCommonInfo::AT_AnyX86NoCfCheck;	 // "nocf_check"
      }
      break;
    case 's':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ort_call", 8) != 0)
          break;
        return AttributeCommonInfo::AT_MipsShortCall;	 // "short_call"
      case 'w':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "ift_", 4) != 0)
          break;
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ttr", 3) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftAttr;	 // "swift_attr"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ame", 3) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftName;	 // "swift_name"
        }
        break;
      }
      break;
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "se_handle", 9) != 0)
        break;
      return AttributeCommonInfo::AT_UseHandle;	 // "use_handle"
    case 'v':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ctorcall", 8) != 0)
          break;
        return AttributeCommonInfo::AT_VectorCall;	 // "vectorcall"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sibility", 8) != 0)
          break;
        return AttributeCommonInfo::AT_Visibility;	 // "visibility"
      }
      break;
    }
    break;
  case 11:	 // 29 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 2 strings to match.
      if (Name[1] != 'l')
        break;
      switch (Name[2]) {
      default: break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+3, "gn_value", 8) != 0)
          break;
        return AttributeCommonInfo::AT_AlignValue;	 // "align_value"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+3, "oc_align", 8) != 0)
          break;
        return AttributeCommonInfo::AT_AllocAlign;	 // "alloc_align"
      }
      break;
    case 'c':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "lled_once", 9) != 0)
          break;
        return AttributeCommonInfo::AT_CalledOnce;	 // "called_once"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_consumed", 9) != 0)
          break;
        return AttributeCommonInfo::AT_CFConsumed;	 // "cf_consumed"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "nstructor", 9) != 0)
          break;
        return AttributeCommonInfo::AT_Constructor;	 // "constructor"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iagnose_if", 10) != 0)
        break;
      return AttributeCommonInfo::AT_DiagnoseIf;	 // "diagnose_if"
    case 'e':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+2, "force_tcb", 9) != 0)
          break;
        return AttributeCommonInfo::AT_EnforceTCB;	 // "enforce_tcb"
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+2, "port_name", 9) != 0)
          break;
        return AttributeCommonInfo::AT_WebAssemblyExportName;	 // "export_name"
      }
      break;
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "allthrough", 10) != 0)
        break;
      return AttributeCommonInfo::AT_FallThrough;	 // "fallthrough"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uarded_var", 10) != 0)
        break;
      return AttributeCommonInfo::AT_GuardedVar;	 // "guarded_var"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mport_name", 10) != 0)
        break;
      return AttributeCommonInfo::AT_WebAssemblyImportName;	 // "import_name"
    case 'm':	 // 2 strings to match.
      if (Name[1] != 'a')
        break;
      switch (Name[2]) {
      default: break;
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+3, "rix_type", 8) != 0)
          break;
        return AttributeCommonInfo::AT_MatrixType;	 // "matrix_type"
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+3, "be_undef", 8) != 0)
          break;
        return AttributeCommonInfo::AT_MaybeUndef;	 // "maybe_undef"
      }
      break;
    case 'n':	 // 5 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 4 strings to match.
        switch (Name[2]) {
        default: break;
        case '_':	 // 1 string to match.
          if (memcmp(Name.data()+3, "sanitize", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitize;	 // "no_sanitize"
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+3, "uplicate", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoDuplicate;	 // "noduplicate"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+3, "icromips", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoMicroMips;	 // "nomicromips"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+3, "blocking", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NonBlocking;	 // "nonblocking"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_consumed", 9) != 0)
          break;
        return AttributeCommonInfo::AT_NSConsumed;	 // "ns_consumed"
      }
      break;
    case 'o':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "jc_", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+6, "ridge", 5) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCBridge;	 // "objc_bridge"
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+6, "irect", 5) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDirect;	 // "objc_direct"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_consumed", 9) != 0)
          break;
        return AttributeCommonInfo::AT_OSConsumed;	 // "os_consumed"
      }
      break;
    case 's':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'w':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "ift_", 4) != 0)
          break;
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "sync", 4) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftAsync;	 // "swift_async"
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+7, "rror", 4) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftError;	 // "swift_error"
        }
        break;
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cl_kernel", 9) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLKernel;	 // "sycl_kernel"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rivial_abi", 10) != 0)
        break;
      return AttributeCommonInfo::AT_TrivialABI;	 // "trivial_abi"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "navailable", 10) != 0)
        break;
      return AttributeCommonInfo::AT_Unavailable;	 // "unavailable"
    case 'v':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ector_size", 10) != 0)
        break;
      return AttributeCommonInfo::AT_VectorSize;	 // "vector_size"
    case 'w':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rn_unused", 9) != 0)
          break;
        return AttributeCommonInfo::AT_WarnUnused;	 // "warn_unused"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ak_import", 9) != 0)
          break;
        return AttributeCommonInfo::AT_WeakImport;	 // "weak_import"
      }
      break;
    }
    break;
  case 12:	 // 15 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "vailability", 11) != 0)
        break;
      return AttributeCommonInfo::AT_Availability;	 // "availability"
    case 'b':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "f_fastcall", 10) != 0)
          break;
        return AttributeCommonInfo::AT_BPFFastCall;	 // "bpf_fastcall"
      case 't':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "f_", 2) != 0)
          break;
        switch (Name[4]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+5, "ecl_tag", 7) != 0)
            break;
          return AttributeCommonInfo::AT_BTFDeclTag;	 // "btf_decl_tag"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+5, "ype_tag", 7) != 0)
            break;
          return AttributeCommonInfo::AT_BTFTypeTag;	 // "btf_type_tag"
        }
        break;
      }
      break;
    case 'c':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ro_wrapper", 10) != 0)
          break;
        return AttributeCommonInfo::AT_CoroWrapper;	 // "coro_wrapper"
      case 'p':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "u_", 2) != 0)
          break;
        switch (Name[4]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+5, "ispatch", 7) != 0)
            break;
          return AttributeCommonInfo::AT_CPUDispatch;	 // "cpu_dispatch"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+5, "pecific", 7) != 0)
            break;
          return AttributeCommonInfo::AT_CPUSpecific;	 // "cpu_specific"
        }
        break;
      }
      break;
    case 'n':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "convergent", 10) != 0)
          break;
        return AttributeCommonInfo::AT_NoConvergent;	 // "noconvergent"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ptx_kernel", 10) != 0)
          break;
        return AttributeCommonInfo::AT_NVPTXKernel;	 // "nvptx_kernel"
      }
      break;
    case 'o':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+2, "jc_boxable", 10) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCBoxable;	 // "objc_boxable"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "encl_local", 10) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLLocalAddressSpace;	 // "opencl_local"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+2, "erloadable", 10) != 0)
          break;
        return AttributeCommonInfo::AT_Overloadable;	 // "overloadable"
      }
      break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "reserve_all", 11) != 0)
        break;
      return AttributeCommonInfo::AT_PreserveAll;	 // "preserve_all"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iscv_vls_cc", 11) != 0)
        break;
      return AttributeCommonInfo::AT_RISCVVLSCC;	 // "riscv_vls_cc"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "wift_bridge", 11) != 0)
        break;
      return AttributeCommonInfo::AT_SwiftBridge;	 // "swift_bridge"
    }
    break;
  case 13:	 // 26 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "dress_space", 11) != 0)
          break;
        return AttributeCommonInfo::AT_AddressSpace;	 // "address_space"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ways_inline", 11) != 0)
          break;
        return AttributeCommonInfo::AT_AlwaysInline;	 // "always_inline"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "dgpu_kernel", 11) != 0)
          break;
        return AttributeCommonInfo::AT_AMDGPUKernelCall;	 // "amdgpu_kernel"
      }
      break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "allable_when", 12) != 0)
        break;
      return AttributeCommonInfo::AT_CallableWhen;	 // "callable_when"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rid_constant", 12) != 0)
        break;
      return AttributeCommonInfo::AT_CUDAGridConstant;	 // "grid_constant"
    case 'i':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "port_module", 11) != 0)
          break;
        return AttributeCommonInfo::AT_WebAssemblyImportModule;	 // "import_module"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+2, "it_priority", 11) != 0)
          break;
        return AttributeCommonInfo::AT_InitPriority;	 // "init_priority"
      }
      break;
    case 'l':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "unch_bounds", 11) != 0)
          break;
        return AttributeCommonInfo::AT_CUDALaunchBounds;	 // "launch_bounds"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "fetimebound", 11) != 0)
          break;
        return AttributeCommonInfo::AT_LifetimeBound;	 // "lifetimebound"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ck_returned", 11) != 0)
          break;
        return AttributeCommonInfo::AT_LockReturned;	 // "lock_returned"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "onallocating", 12) != 0)
        break;
      return AttributeCommonInfo::AT_NonAllocating;	 // "nonallocating"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "pencl_global", 12) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLGlobalAddressSpace;	 // "opencl_global"
    case 'p':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'r':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "eserve_", 7) != 0)
          break;
        switch (Name[9]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+10, "ost", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreserveMost;	 // "preserve_most"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+10, "one", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreserveNone;	 // "preserve_none"
        }
        break;
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_guarded_by", 11) != 0)
          break;
        return AttributeCommonInfo::AT_PtGuardedBy;	 // "pt_guarded_by"
      }
      break;
    case 'r':	 // 2 strings to match.
      if (Name[1] != 'e')
        break;
      switch (Name[2]) {
      default: break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+3, "nitializes", 10) != 0)
          break;
        return AttributeCommonInfo::AT_Reinitializes;	 // "reinitializes"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+3, "urns_twice", 10) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnsTwice;	 // "returns_twice"
      }
      break;
    case 's':	 // 5 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "t_typestate", 11) != 0)
          break;
        return AttributeCommonInfo::AT_SetTypestate;	 // "set_typestate"
      case 'w':	 // 4 strings to match.
        if (memcmp(Name.data()+2, "ift_", 4) != 0)
          break;
        switch (Name[6]) {
        default: break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ontext", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftContext;	 // "swift_context"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ewtype", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftNewType;	 // "swift_newtype"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+7, "rivate", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftPrivate;	 // "swift_private"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+7, "rapper", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftNewType;	 // "swift_wrapper"
        }
        break;
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "arget_clones", 12) != 0)
        break;
      return AttributeCommonInfo::AT_TargetClones;	 // "target_clones"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ninitialized", 12) != 0)
        break;
      return AttributeCommonInfo::AT_Uninitialized;	 // "uninitialized"
    case 'v':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ec_type_hint", 12) != 0)
        break;
      return AttributeCommonInfo::AT_VecTypeHint;	 // "vec_type_hint"
    case 'x':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ray_log_args", 12) != 0)
        break;
      return AttributeCommonInfo::AT_XRayLogArgs;	 // "xray_log_args"
    }
    break;
  case 14:	 // 21 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "quire", 5) != 0)
          break;
        switch (Name[7]) {
        default: break;
        case '_':	 // 1 string to match.
          if (memcmp(Name.data()+8, "handle", 6) != 0)
            break;
          return AttributeCommonInfo::AT_AcquireHandle;	 // "acquire_handle"
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+8, "_after", 6) != 0)
            break;
          return AttributeCommonInfo::AT_AcquiredAfter;	 // "acquired_after"
        }
        break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ways_destroy", 12) != 0)
          break;
        return AttributeCommonInfo::AT_AlwaysDestroy;	 // "always_destroy"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sume_aligned", 12) != 0)
          break;
        return AttributeCommonInfo::AT_AssumeAligned;	 // "assume_aligned"
      }
      break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "udart_builtin", 13) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "cudart_builtin"
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "evice_builtin", 13) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "device_builtin"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ormat_matches", 13) != 0)
        break;
      return AttributeCommonInfo::AT_FormatMatches;	 // "format_matches"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ntel_ocl_bicc", 13) != 0)
        break;
      return AttributeCommonInfo::AT_IntelOclBicc;	 // "intel_ocl_bicc"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ocks_excluded", 13) != 0)
        break;
      return AttributeCommonInfo::AT_LocksExcluded;	 // "locks_excluded"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "o_split_stack", 13) != 0)
        break;
      return AttributeCommonInfo::AT_NoSplitStack;	 // "no_split_stack"
    case 'o':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "jc_", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+6, "xception", 8) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCException;	 // "objc_exception"
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+6, "wnership", 8) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCOwnership;	 // "objc_ownership"
        }
        break;
      case 'p':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "encl_", 5) != 0)
          break;
        switch (Name[7]) {
        default: break;
        case 'g':	 // 1 string to match.
          if (memcmp(Name.data()+8, "eneric", 6) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLGenericAddressSpace;	 // "opencl_generic"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+8, "rivate", 6) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLPrivateAddressSpace;	 // "opencl_private"
        }
        break;
      }
      break;
    case 'p':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'r':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "eferred_", 8) != 0)
          break;
        switch (Name[10]) {
        default: break;
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+11, "ame", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreferredName;	 // "preferred_name"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+11, "ype", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreferredType;	 // "preferred_type"
        }
        break;
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_guarded_var", 12) != 0)
          break;
        return AttributeCommonInfo::AT_PtGuardedVar;	 // "pt_guarded_var"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "elease_handle", 13) != 0)
        break;
      return AttributeCommonInfo::AT_ReleaseHandle;	 // "release_handle"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "wiftasynccall", 13) != 0)
        break;
      return AttributeCommonInfo::AT_SwiftAsyncCall;	 // "swiftasynccall"
    case 't':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rget_version", 12) != 0)
          break;
        return AttributeCommonInfo::AT_TargetVersion;	 // "target_version"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "st_typestate", 12) != 0)
          break;
        return AttributeCommonInfo::AT_TestTypestate;	 // "test_typestate"
      }
      break;
    }
    break;
  case 15:	 // 20 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rch64_sve_pcs", 13) != 0)
          break;
        return AttributeCommonInfo::AT_AArch64SVEPcs;	 // "aarch64_sve_pcs"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "quired_before", 13) != 0)
          break;
        return AttributeCommonInfo::AT_AcquiredBefore;	 // "acquired_before"
      case 'm':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "dgpu_num_", 9) != 0)
          break;
        switch (Name[11]) {
        default: break;
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+12, "gpr", 3) != 0)
            break;
          return AttributeCommonInfo::AT_AMDGPUNumSGPR;	 // "amdgpu_num_sgpr"
        case 'v':	 // 1 string to match.
          if (memcmp(Name.data()+12, "gpr", 3) != 0)
            break;
          return AttributeCommonInfo::AT_AMDGPUNumVGPR;	 // "amdgpu_num_vgpr"
        }
        break;
      }
      break;
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "xt_vector_type", 14) != 0)
        break;
      return AttributeCommonInfo::AT_ExtVectorType;	 // "ext_vector_type"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "unction_return", 14) != 0)
        break;
      return AttributeCommonInfo::AT_FunctionReturnThunks;	 // "function_return"
    case 'n':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "t_tail_called", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NotTailCalled;	 // "not_tail_called"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_error_domain", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NSErrorDomain;	 // "ns_error_domain"
      }
      break;
    case 'o':	 // 5 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "jc_", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+6, "lass_stub", 9) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCClassStub;	 // "objc_class_stub"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+6, "oot_class", 9) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRootClass;	 // "objc_root_class"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "encl_constant", 13) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLConstantAddressSpace;	 // "opencl_constant"
      case 'w':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "nership_", 8) != 0)
          break;
        switch (Name[10]) {
        default: break;
        case 'h':	 // 1 string to match.
          if (memcmp(Name.data()+11, "olds", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Ownership;	 // "ownership_holds"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+11, "akes", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Ownership;	 // "ownership_takes"
        }
        break;
      }
      break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "aram_typestate", 14) != 0)
        break;
      return AttributeCommonInfo::AT_ParamTypestate;	 // "param_typestate"
    case 'r':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "turns_nonnull", 13) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnsNonNull;	 // "returns_nonnull"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "scv_vector_cc", 13) != 0)
          break;
        return AttributeCommonInfo::AT_RISCVVectorCC;	 // "riscv_vector_cc"
      }
      break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "coped_lockable", 14) != 0)
        break;
      return AttributeCommonInfo::AT_ScopedLockable;	 // "scoped_lockable"
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ype_visibility", 14) != 0)
        break;
      return AttributeCommonInfo::AT_TypeVisibility;	 // "type_visibility"
    case 'u':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+2, "lock_function", 13) != 0)
          break;
        return AttributeCommonInfo::AT_ReleaseCapability;	 // "unlock_function"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ing_if_exists", 13) != 0)
          break;
        return AttributeCommonInfo::AT_UsingIfExists;	 // "using_if_exists"
      }
      break;
    }
    break;
  case 16:	 // 14 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oro_return_type", 15) != 0)
        break;
      return AttributeCommonInfo::AT_CoroReturnType;	 // "coro_return_type"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nforce_tcb_leaf", 15) != 0)
        break;
      return AttributeCommonInfo::AT_EnforceTCBLeaf;	 // "enforce_tcb_leaf"
    case 'h':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ybrid_patchable", 15) != 0)
        break;
      return AttributeCommonInfo::AT_HybridPatchable;	 // "hybrid_patchable"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nternal_linkage", 15) != 0)
        break;
      return AttributeCommonInfo::AT_InternalLinkage;	 // "internal_linkage"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "in_vector_width", 15) != 0)
        break;
      return AttributeCommonInfo::AT_MinVectorWidth;	 // "min_vector_width"
    case 'n':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "on_vector_type", 14) != 0)
          break;
        return AttributeCommonInfo::AT_NeonVectorType;	 // "neon_vector_type"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_consumes_self", 14) != 0)
          break;
        return AttributeCommonInfo::AT_NSConsumesSelf;	 // "ns_consumes_self"
      }
      break;
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "s_consumes_this", 15) != 0)
        break;
      return AttributeCommonInfo::AT_OSConsumesThis;	 // "os_consumes_this"
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ass_object_size", 15) != 0)
        break;
      return AttributeCommonInfo::AT_PassObjectSize;	 // "pass_object_size"
    case 'r':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ndomize_layout", 14) != 0)
          break;
        return AttributeCommonInfo::AT_RandomizeLayout;	 // "randomize_layout"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "turn_typestate", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnTypestate;	 // "return_typestate"
      }
      break;
    case 's':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "zed_by_or_null", 14) != 0)
          break;
        return AttributeCommonInfo::AT_SizedByOrNull;	 // "sized_by_or_null"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "andalone_debug", 14) != 0)
          break;
        return AttributeCommonInfo::AT_StandaloneDebug;	 // "standalone_debug"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ift_async_name", 14) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftAsyncName;	 // "swift_async_name"
      }
      break;
    }
    break;
  case 17:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+2, "alyzer_noreturn", 15) != 0)
          break;
        return AttributeCommonInfo::AT_AnalyzerNoReturn;	 // "analyzer_noreturn"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sert_capability", 15) != 0)
          break;
        return AttributeCommonInfo::AT_AssertCapability;	 // "assert_capability"
      }
      break;
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nterrupt_save_fp", 16) != 0)
        break;
      return AttributeCommonInfo::AT_ARMInterruptSaveFP;	 // "interrupt_save_fp"
    case 'o':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+2, "jc_runtime_name", 15) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCRuntimeName;	 // "objc_runtime_name"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+2, "nership_returns", 15) != 0)
          break;
        return AttributeCommonInfo::AT_Ownership;	 // "ownership_returns"
      }
      break;
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ared_capability", 15) != 0)
          break;
        return AttributeCommonInfo::AT_Capability;	 // "shared_capability"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ift_async_error", 15) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftAsyncError;	 // "swift_async_error"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ransparent_union", 16) != 0)
        break;
      return AttributeCommonInfo::AT_TransparentUnion;	 // "transparent_union"
    }
    break;
  case 18:	 // 23 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rch64_vector_pcs", 16) != 0)
          break;
        return AttributeCommonInfo::AT_AArch64VectorPcs;	 // "aarch64_vector_pcs"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "quire_capability", 16) != 0)
          break;
        return AttributeCommonInfo::AT_AcquireCapability;	 // "acquire_capability"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "sert_shared_lock", 16) != 0)
          break;
        return AttributeCommonInfo::AT_AssertCapability;	 // "assert_shared_lock"
      }
      break;
    case 'c':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rries_dependency", 16) != 0)
          break;
        return AttributeCommonInfo::AT_CarriesDependency;	 // "carries_dependency"
      case 'o':	 // 2 strings to match.
        switch (Name[2]) {
        default: break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+3, "o_lifetimebound", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CoroLifetimeBound;	 // "coro_lifetimebound"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+3, "nted_by_or_null", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CountedByOrNull;	 // "counted_by_or_null"
        }
        break;
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "isable_tail_calls", 17) != 0)
        break;
      return AttributeCommonInfo::AT_DisableTailCalls;	 // "disable_tail_calls"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "num_extensibility", 17) != 0)
        break;
      return AttributeCommonInfo::AT_EnumExtensibility;	 // "enum_extensibility"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "boutletcollection", 17) != 0)
        break;
      return AttributeCommonInfo::AT_IBOutletCollection;	 // "iboutletcollection"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ig_server_routine", 17) != 0)
        break;
      return AttributeCommonInfo::AT_MIGServerRoutine;	 // "mig_server_routine"
    case 'n':	 // 4 strings to match.
      if (memcmp(Name.data()+1, "o_s", 3) != 0)
        break;
      switch (Name[4]) {
      default: break;
      case 'a':	 // 2 strings to match.
        if (memcmp(Name.data()+5, "nitize_", 7) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+13, "emory", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "no_sanitize_memory"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+13, "hread", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "no_sanitize_thread"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+5, "ecializations", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoSpecializations;	 // "no_specializations"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+5, "ack_protector", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoStackProtector;	 // "no_stack_protector"
      }
      break;
    case 'o':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "jc_", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+6, "ethod_family", 12) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCMethodFamily;	 // "objc_method_family"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+6, "onlazy_class", 12) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCNonLazyClass;	 // "objc_nonlazy_class"
        }
        break;
      case 'p':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "encl_", 5) != 0)
          break;
        switch (Name[7]) {
        default: break;
        case 'g':	 // 1 string to match.
          if (memcmp(Name.data()+8, "lobal_host", 10) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLGlobalHostAddressSpace;	 // "opencl_global_host"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+8, "nroll_hint", 10) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLUnrollHint;	 // "opencl_unroll_hint"
        }
        break;
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "elease_capability", 17) != 0)
        break;
      return AttributeCommonInfo::AT_ReleaseCapability;	 // "release_capability"
    case 's':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'w':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "ift_", 4) != 0)
          break;
        switch (Name[6]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+7, "rror_result", 11) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftErrorResult;	 // "swift_error_result"
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+7, "bjc_members", 11) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftObjCMembers;	 // "swift_objc_members"
        }
        break;
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cl_special_class", 16) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLSpecialClass;	 // "sycl_special_class"
      }
      break;
    case 'w':	 // 1 string to match.
      if (memcmp(Name.data()+1, "arn_unused_result", 17) != 0)
        break;
      return AttributeCommonInfo::AT_WarnUnusedResult;	 // "warn_unused_result"
    }
    break;
  case 19:	 // 22 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "dgpu_waves_per_eu", 17) != 0)
          break;
        return AttributeCommonInfo::AT_AMDGPUWavesPerEU;	 // "amdgpu_waves_per_eu"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "m_sve_vector_bits", 17) != 0)
          break;
        return AttributeCommonInfo::AT_ArmSveVectorBits;	 // "arm_sve_vector_bits"
      }
      break;
    case 'c':	 // 6 strings to match.
      switch (Name[1]) {
      default: break;
      case 'f':	 // 3 strings to match.
        if (Name[2] != '_')
          break;
        switch (Name[3]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+4, "udited_transfer", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFAuditedTransfer;	 // "cf_audited_transfer"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+4, "eturns_retained", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFReturnsRetained;	 // "cf_returns_retained"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+4, "nknown_transfer", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFUnknownTransfer;	 // "cf_unknown_transfer"
        }
        break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ang_builtin_alias", 17) != 0)
          break;
        return AttributeCommonInfo::AT_BuiltinAlias;	 // "clang_builtin_alias"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "se_nonsecure_call", 17) != 0)
          break;
        return AttributeCommonInfo::AT_CmseNSCall;	 // "cmse_nonsecure_call"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ro_await_elidable", 17) != 0)
          break;
        return AttributeCommonInfo::AT_CoroAwaitElidable;	 // "coro_await_elidable"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iagnose_as_builtin", 18) != 0)
        break;
      return AttributeCommonInfo::AT_DiagnoseAsBuiltin;	 // "diagnose_as_builtin"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ifetime_capture_by", 18) != 0)
        break;
      return AttributeCommonInfo::AT_LifetimeCaptureBy;	 // "lifetime_capture_by"
    case 'n':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 2 strings to match.
        if (Name[2] != '_')
          break;
        switch (Name[3]) {
        default: break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+4, "andomize_layout", 15) != 0)
            break;
          return AttributeCommonInfo::AT_NoRandomizeLayout;	 // "no_randomize_layout"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+4, "anitize_address", 15) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "no_sanitize_address"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_returns_retained", 17) != 0)
          break;
        return AttributeCommonInfo::AT_NSReturnsRetained;	 // "ns_returns_retained"
      }
      break;
    case 'o':	 // 5 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 4 strings to match.
        if (memcmp(Name.data()+2, "jc_", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+6, "ridge_", 6) != 0)
            break;
          switch (Name[12]) {
          default: break;
          case 'm':	 // 1 string to match.
            if (memcmp(Name.data()+13, "utable", 6) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridgeMutable;	 // "objc_bridge_mutable"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+13, "elated", 6) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridgeRelated;	 // "objc_bridge_related"
          }
          break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+6, "irect_members", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDirectMembers;	 // "objc_direct_members"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+6, "equires_super", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRequiresSuper;	 // "objc_requires_super"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_returns_retained", 17) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsRetained;	 // "os_returns_retained"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "equires_capability", 18) != 0)
        break;
      return AttributeCommonInfo::AT_RequiresCapability;	 // "requires_capability"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "wift_async_context", 18) != 0)
        break;
      return AttributeCommonInfo::AT_SwiftAsyncContext;	 // "swift_async_context"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nsafe_buffer_usage", 18) != 0)
        break;
      return AttributeCommonInfo::AT_UnsafeBufferUsage;	 // "unsafe_buffer_usage"
    case 'z':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ero_call_used_regs", 18) != 0)
        break;
      return AttributeCommonInfo::AT_ZeroCallUsedRegs;	 // "zero_call_used_regs"
    }
    break;
  case 20:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+2, "spv_libclc_builtin", 18) != 0)
          break;
        return AttributeCommonInfo::AT_ClspvLibclcBuiltin;	 // "clspv_libclc_builtin"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "se_nonsecure_entry", 18) != 0)
          break;
        return AttributeCommonInfo::AT_CmseNSEntry;	 // "cmse_nonsecure_entry"
      }
      break;
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oader_uninitialized", 19) != 0)
        break;
      return AttributeCommonInfo::AT_LoaderUninitialized;	 // "loader_uninitialized"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eon_polyvector_type", 19) != 0)
        break;
      return AttributeCommonInfo::AT_NeonPolyVectorType;	 // "neon_polyvector_type"
    case 'o':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+2, "jc_runtime_visible", 18) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCRuntimeVisible;	 // "objc_runtime_visible"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "encl_global_device", 18) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLGlobalDeviceAddressSpace;	 // "opencl_global_device"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eqd_work_group_size", 19) != 0)
        break;
      return AttributeCommonInfo::AT_ReqdWorkGroupSize;	 // "reqd_work_group_size"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "hared_lock_function", 19) != 0)
        break;
      return AttributeCommonInfo::AT_AcquireCapability;	 // "shared_lock_function"
    case 'w':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ork_group_size_hint", 19) != 0)
        break;
      return AttributeCommonInfo::AT_WorkGroupSizeHint;	 // "work_group_size_hint"
    }
    break;
  case 21:	 // 11 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ssert_exclusive_lock", 20) != 0)
        break;
      return AttributeCommonInfo::AT_AssertCapability;	 // "assert_exclusive_lock"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "to_visibility_public", 20) != 0)
        break;
      return AttributeCommonInfo::AT_LTOVisibilityPublic;	 // "lto_visibility_public"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bjc_precise_lifetime", 20) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCPreciseLifetime;	 // "objc_precise_lifetime"
    case 'p':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "inter_with_type_tag", 19) != 0)
          break;
        return AttributeCommonInfo::AT_ArgumentWithTypeTag;	 // "pointer_with_type_tag"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "eserve_access_index", 19) != 0)
          break;
        return AttributeCommonInfo::AT_BPFPreserveAccessIndex;	 // "preserve_access_index"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iscv_rvv_vector_bits", 20) != 0)
        break;
      return AttributeCommonInfo::AT_RISCVRVVVectorBits;	 // "riscv_rvv_vector_bits"
    case 's':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ared_locks_required", 19) != 0)
          break;
        return AttributeCommonInfo::AT_RequiresCapability;	 // "shared_locks_required"
      case 'w':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "ift_", 4) != 0)
          break;
        switch (Name[6]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ridged_typedef", 14) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftBridgedTypedef;	 // "swift_bridged_typedef"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ndirect_result", 14) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftIndirectResult;	 // "swift_indirect_result"
        }
        break;
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ype_tag_for_datatype", 20) != 0)
        break;
      return AttributeCommonInfo::AT_TypeTagForDatatype;	 // "type_tag_for_datatype"
    case 'x':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ray_never_instrument", 20) != 0)
        break;
      return AttributeCommonInfo::AT_XRayInstrument;	 // "xray_never_instrument"
    }
    break;
  case 22:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rgument_with_type_tag", 21) != 0)
        break;
      return AttributeCommonInfo::AT_ArgumentWithTypeTag;	 // "argument_with_type_tag"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "xternal_source_symbol", 21) != 0)
        break;
      return AttributeCommonInfo::AT_ExternalSourceSymbol;	 // "external_source_symbol"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "o_instrument_function", 21) != 0)
        break;
      return AttributeCommonInfo::AT_NoInstrumentFunction;	 // "no_instrument_function"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bjc_independent_class", 21) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCIndependentClass;	 // "objc_independent_class"
    case 'p':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "eserve_static_offset", 20) != 0)
          break;
        return AttributeCommonInfo::AT_BPFPreserveStaticOffset;	 // "preserve_static_offset"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rauth_vtable_pointer", 20) != 0)
          break;
        return AttributeCommonInfo::AT_VTablePointerAuthentication;	 // "ptrauth_vtable_pointer"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ry_acquire_capability", 21) != 0)
        break;
      return AttributeCommonInfo::AT_TryAcquireCapability;	 // "try_acquire_capability"
    case 'x':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ray_always_instrument", 21) != 0)
        break;
      return AttributeCommonInfo::AT_XRayInstrument;	 // "xray_always_instrument"
    }
    break;
  case 23:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "f_returns_not_retained", 22) != 0)
        break;
      return AttributeCommonInfo::AT_CFReturnsNotRetained;	 // "cf_returns_not_retained"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "xclusive_lock_function", 22) != 0)
        break;
      return AttributeCommonInfo::AT_AcquireCapability;	 // "exclusive_lock_function"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "orce_align_arg_pointer", 22) != 0)
        break;
      return AttributeCommonInfo::AT_X86ForceAlignArgPointer;	 // "force_align_arg_pointer"
    case 'n':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "s_returns_", 10) != 0)
        break;
      switch (Name[11]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+12, "utoreleased", 11) != 0)
          break;
        return AttributeCommonInfo::AT_NSReturnsAutoreleased;	 // "ns_returns_autoreleased"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+12, "ot_retained", 11) != 0)
          break;
        return AttributeCommonInfo::AT_NSReturnsNotRetained;	 // "ns_returns_not_retained"
      }
      break;
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "s_returns_not_retained", 22) != 0)
        break;
      return AttributeCommonInfo::AT_OSReturnsNotRetained;	 // "os_returns_not_retained"
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ared_trylock_function", 21) != 0)
          break;
        return AttributeCommonInfo::AT_TryAcquireCapability;	 // "shared_trylock_function"
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cl_kernel_entry_point", 21) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLKernelEntryPoint;	 // "sycl_kernel_entry_point"
      }
      break;
    }
    break;
  case 24:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ssert_shared_capability", 23) != 0)
        break;
      return AttributeCommonInfo::AT_AssertCapability;	 // "assert_shared_capability"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "fi_canonical_jump_table", 23) != 0)
        break;
      return AttributeCommonInfo::AT_CFICanonicalJumpTable;	 // "cfi_canonical_jump_table"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "xclusive_locks_required", 23) != 0)
        break;
      return AttributeCommonInfo::AT_RequiresCapability;	 // "exclusive_locks_required"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bjc_externally_retained", 23) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCExternallyRetained;	 // "objc_externally_retained"
    case 'p':	 // 2 strings to match.
      if (Name[1] != 'a')
        break;
      switch (Name[2]) {
      default: break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+3, "s_dynamic_object_size", 21) != 0)
          break;
        return AttributeCommonInfo::AT_PassObjectSize;	 // "pass_dynamic_object_size"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+3, "chable_function_entry", 21) != 0)
          break;
        return AttributeCommonInfo::AT_PatchableFunctionEntry;	 // "patchable_function_entry"
      }
      break;
    }
    break;
  case 25:	 // 7 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_clang_arm_builtin_alias", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ArmBuiltinAlias;	 // "__clang_arm_builtin_alias"
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "cquire_shared_capability", 24) != 0)
        break;
      return AttributeCommonInfo::AT_AcquireCapability;	 // "acquire_shared_capability"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ntel_reqd_sub_group_size", 24) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLIntelReqdSubGroupSize;	 // "intel_reqd_sub_group_size"
    case 'n':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "o_", 2) != 0)
        break;
      switch (Name[3]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+4, "aller_saved_registers", 21) != 0)
          break;
        return AttributeCommonInfo::AT_AnyX86NoCallerSavedRegisters;	 // "no_caller_saved_registers"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+4, "hread_safety_analysis", 21) != 0)
          break;
        return AttributeCommonInfo::AT_NoThreadSafetyAnalysis;	 // "no_thread_safety_analysis"
      }
      break;
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bjc_non_runtime_protocol", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCNonRuntimeProtocol;	 // "objc_non_runtime_protocol"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "elease_shared_capability", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ReleaseCapability;	 // "release_shared_capability"
    }
    break;
  case 26:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mdgpu_max_num_work_groups", 25) != 0)
        break;
      return AttributeCommonInfo::AT_AMDGPUMaxNumWorkGroups;	 // "amdgpu_max_num_work_groups"
    case 'c':	 // 2 strings to match.
      if (Name[1] != 'o')
        break;
      switch (Name[2]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+3, "sumable_auto_cast_state", 23) != 0)
          break;
        return AttributeCommonInfo::AT_ConsumableAutoCast;	 // "consumable_auto_cast_state"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+3, "o_disable_lifetimebound", 23) != 0)
          break;
        return AttributeCommonInfo::AT_CoroDisableLifetimeBound;	 // "coro_disable_lifetimebound"
      }
      break;
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "xclusive_trylock_function", 25) != 0)
        break;
      return AttributeCommonInfo::AT_TryAcquireCapability;	 // "exclusive_trylock_function"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "o_address_safety_analysis", 25) != 0)
        break;
      return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "no_address_safety_analysis"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bjc_returns_inner_pointer", 25) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCReturnsInnerPointer;	 // "objc_returns_inner_pointer"
    case 'r':	 // 2 strings to match.
      if (Name[1] != 'e')
        break;
      switch (Name[2]) {
      default: break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+3, "ease_generic_capability", 23) != 0)
          break;
        return AttributeCommonInfo::AT_ReleaseCapability;	 // "release_generic_capability"
      case 'q':	 // 1 string to match.
        if (memcmp(Name.data()+3, "uires_shared_capability", 23) != 0)
          break;
        return AttributeCommonInfo::AT_RequiresCapability;	 // "requires_shared_capability"
      }
      break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "peculative_load_hardening", 25) != 0)
        break;
      return AttributeCommonInfo::AT_SpeculativeLoadHardening;	 // "speculative_load_hardening"
    }
    break;
  case 27:	 // 7 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mdgpu_flat_work_group_size", 26) != 0)
        break;
      return AttributeCommonInfo::AT_AMDGPUFlatWorkGroupSize;	 // "amdgpu_flat_work_group_size"
    case 'd':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "evice_builtin_", 14) != 0)
        break;
      switch (Name[15]) {
      default: break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+16, "urface_type", 11) != 0)
          break;
        return AttributeCommonInfo::AT_CUDADeviceBuiltinSurfaceType;	 // "device_builtin_surface_type"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+16, "exture_type", 11) != 0)
          break;
        return AttributeCommonInfo::AT_CUDADeviceBuiltinTextureType;	 // "device_builtin_texture_type"
      }
      break;
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nforce_read_only_placement", 26) != 0)
        break;
      return AttributeCommonInfo::AT_ReadOnlyPlacement;	 // "enforce_read_only_placement"
    case 'o':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+2, "jc_", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+6, "esignated_initializer", 21) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDesignatedInitializer;	 // "objc_designated_initializer"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+6, "ubclassing_restricted", 21) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCSubclassingRestricted;	 // "objc_subclassing_restricted"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "_returns_retained_on_zero", 25) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsRetainedOnZero;	 // "os_returns_retained_on_zero"
      }
      break;
    }
    break;
  case 28:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "co", 2) != 0)
      break;
    switch (Name[2]) {
    default: break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+3, "sumable_set_state_on_read", 25) != 0)
        break;
      return AttributeCommonInfo::AT_ConsumableSetOnRead;	 // "consumable_set_state_on_read"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+3, "o_await_elidable_argument", 25) != 0)
        break;
      return AttributeCommonInfo::AT_CoroAwaitElidableArgument;	 // "coro_await_elidable_argument"
    }
    break;
  case 29:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "o_speculative_load_hardening", 28) != 0)
        break;
      return AttributeCommonInfo::AT_NoSpeculativeLoadHardening;	 // "no_speculative_load_hardening"
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ry_acquire_shared_capability", 28) != 0)
        break;
      return AttributeCommonInfo::AT_TryAcquireCapability;	 // "try_acquire_shared_capability"
    }
    break;
  case 30:	 // 1 string to match.
    if (memcmp(Name.data()+0, "no_profile_instrument_function", 30) != 0)
      break;
    return AttributeCommonInfo::AT_NoProfileFunction;	 // "no_profile_instrument_function"
  case 31:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oro_only_destroy_when_complete", 30) != 0)
        break;
      return AttributeCommonInfo::AT_CoroOnlyDestroyWhenComplete;	 // "coro_only_destroy_when_complete"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "s_returns_retained_on_non_zero", 30) != 0)
        break;
      return AttributeCommonInfo::AT_OSReturnsRetainedOnNonZero;	 // "os_returns_retained_on_non_zero"
    case 'r':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "equire_", 7) != 0)
        break;
      switch (Name[8]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+9, "onstant_initialization", 22) != 0)
          break;
        return AttributeCommonInfo::AT_ConstInit;	 // "require_constant_initialization"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+9, "xplicit_initialization", 22) != 0)
          break;
        return AttributeCommonInfo::AT_ExplicitInit;	 // "require_explicit_initialization"
      }
      break;
    }
    break;
  case 33:	 // 1 string to match.
    if (memcmp(Name.data()+0, "disable_sanitizer_instrumentation", 33) != 0)
      break;
    return AttributeCommonInfo::AT_DisableSanitizerInstrumentation;	 // "disable_sanitizer_instrumentation"
  case 34:	 // 1 string to match.
    if (memcmp(Name.data()+0, "objc_requires_property_definitions", 34) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCRequiresPropertyDefs;	 // "objc_requires_property_definitions"
  case 35:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_clang_arm_mve_strict_polymorphism", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ArmMveStrictPolymorphism;	 // "__clang_arm_mve_strict_polymorphism"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "xclude_from_explicit_instantiation", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ExcludeFromExplicitInstantiation;	 // "exclude_from_explicit_instantiation"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bjc_arc_weak_reference_unavailable", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ArcWeakrefUnavailable;	 // "objc_arc_weak_reference_unavailable"
    }
    break;
  case 37:	 // 1 string to match.
    if (memcmp(Name.data()+0, "available_only_in_default_eval_method", 37) != 0)
      break;
    return AttributeCommonInfo::AT_AvailableOnlyInDefaultEvalMethod;	 // "available_only_in_default_eval_method"
  case 46:	 // 1 string to match.
    if (memcmp(Name.data()+0, "objc_protocol_requires_explicit_implementation", 46) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCExplicitProtocolImpl;	 // "objc_protocol_requires_explicit_implementation"
  }
  } else if (AttributeCommonInfo::AS_Declspec == Syntax) {
  switch (Name.size()) {
  default: break;
  case 4:	 // 1 string to match.
    if (memcmp(Name.data()+0, "uuid", 4) != 0)
      break;
    return AttributeCommonInfo::AT_Uuid;	 // "uuid"
  case 5:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lign", 4) != 0)
        break;
      return AttributeCommonInfo::AT_Aligned;	 // "align"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uard", 4) != 0)
        break;
      return AttributeCommonInfo::AT_CFGuard;	 // "guard"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "aked", 4) != 0)
        break;
      return AttributeCommonInfo::AT_Naked;	 // "naked"
    }
    break;
  case 6:	 // 1 string to match.
    if (memcmp(Name.data()+0, "thread", 6) != 0)
      break;
    return AttributeCommonInfo::AT_Thread;	 // "thread"
  case 7:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "no", 2) != 0)
      break;
    switch (Name[2]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+3, "lias", 4) != 0)
        break;
      return AttributeCommonInfo::AT_NoAlias;	 // "noalias"
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+3, "hrow", 4) != 0)
        break;
      return AttributeCommonInfo::AT_NoThrow;	 // "nothrow"
    }
    break;
  case 8:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_host__", 7) != 0)
        break;
      return AttributeCommonInfo::AT_CUDAHost;	 // "__host__"
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "llocate", 7) != 0)
        break;
      return AttributeCommonInfo::AT_Section;	 // "allocate"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ode_seg", 7) != 0)
        break;
      return AttributeCommonInfo::AT_CodeSeg;	 // "code_seg"
    case 'n':	 // 3 strings to match.
      if (Name[1] != 'o')
        break;
      switch (Name[2]) {
      default: break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+3, "nline", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoInline;	 // "noinline"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+3, "eturn", 5) != 0)
          break;
        return AttributeCommonInfo::AT_NoReturn;	 // "noreturn"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+3, "table", 5) != 0)
          break;
        return AttributeCommonInfo::AT_MSNoVTable;	 // "novtable"
      }
      break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "roperty", 7) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "property"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "estrict", 7) != 0)
        break;
      return AttributeCommonInfo::AT_Restrict;	 // "restrict"
    }
    break;
  case 9:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "llocator", 8) != 0)
        break;
      return AttributeCommonInfo::AT_MSAllocator;	 // "allocator"
    case 'd':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "ll", 2) != 0)
        break;
      switch (Name[3]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+4, "xport", 5) != 0)
          break;
        return AttributeCommonInfo::AT_DLLExport;	 // "dllexport"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+4, "mport", 5) != 0)
          break;
        return AttributeCommonInfo::AT_DLLImport;	 // "dllimport"
      }
      break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "electany", 8) != 0)
        break;
      return AttributeCommonInfo::AT_SelectAny;	 // "selectany"
    }
    break;
  case 10:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 3 strings to match.
      if (Name[1] != '_')
        break;
      switch (Name[2]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+3, "evice__", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CUDADevice;	 // "__device__"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+3, "lobal__", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CUDAGlobal;	 // "__global__"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+3, "hared__", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CUDAShared;	 // "__shared__"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eprecated", 9) != 0)
        break;
      return AttributeCommonInfo::AT_Deprecated;	 // "deprecated"
    }
    break;
  case 11:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_managed__", 10) != 0)
        break;
      return AttributeCommonInfo::AT_HIPManaged;	 // "__managed__"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mpty_bases", 10) != 0)
        break;
      return AttributeCommonInfo::AT_EmptyBases;	 // "empty_bases"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "o_init_all", 10) != 0)
        break;
      return AttributeCommonInfo::AT_NoTrivialAutoVarInit;	 // "no_init_all"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "afebuffers", 10) != 0)
        break;
      return AttributeCommonInfo::AT_NoStackProtector;	 // "safebuffers"
    }
    break;
  case 12:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_constant__", 11) != 0)
        break;
      return AttributeCommonInfo::AT_CUDAConstant;	 // "__constant__"
    case 'c':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "pu_", 3) != 0)
        break;
      switch (Name[4]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+5, "ispatch", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CPUDispatch;	 // "cpu_dispatch"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+5, "pecific", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CPUSpecific;	 // "cpu_specific"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oconvergent", 11) != 0)
        break;
      return AttributeCommonInfo::AT_NoConvergent;	 // "noconvergent"
    }
    break;
  case 14:	 // 1 string to match.
    if (memcmp(Name.data()+0, "layout_version", 14) != 0)
      break;
    return AttributeCommonInfo::AT_LayoutVersion;	 // "layout_version"
  case 15:	 // 1 string to match.
    if (memcmp(Name.data()+0, "strict_gs_check", 15) != 0)
      break;
    return AttributeCommonInfo::AT_StrictGuardStackCheck;	 // "strict_gs_check"
  case 16:	 // 1 string to match.
    if (memcmp(Name.data()+0, "hybrid_patchable", 16) != 0)
      break;
    return AttributeCommonInfo::AT_HybridPatchable;	 // "hybrid_patchable"
  case 17:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "__", 2) != 0)
      break;
    switch (Name[2]) {
    default: break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+3, "rid_constant__", 14) != 0)
        break;
      return AttributeCommonInfo::AT_CUDAGridConstant;	 // "__grid_constant__"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+3, "aunch_bounds__", 14) != 0)
        break;
      return AttributeCommonInfo::AT_CUDALaunchBounds;	 // "__launch_bounds__"
    }
    break;
  case 18:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "__", 2) != 0)
      break;
    switch (Name[2]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+3, "udart_builtin__", 15) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "__cudart_builtin__"
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+3, "evice_builtin__", 15) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "__device_builtin__"
    }
    break;
  case 31:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "__device_builtin_", 17) != 0)
      break;
    switch (Name[17]) {
    default: break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+18, "urface_type__", 13) != 0)
        break;
      return AttributeCommonInfo::AT_CUDADeviceBuiltinSurfaceType;	 // "__device_builtin_surface_type__"
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+18, "exture_type__", 13) != 0)
        break;
      return AttributeCommonInfo::AT_CUDADeviceBuiltinTextureType;	 // "__device_builtin_texture_type__"
    }
    break;
  }
  } else if (AttributeCommonInfo::AS_Microsoft == Syntax) {
  switch (Name.size()) {
  default: break;
  case 4:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oop", 3) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLLoopHint;	 // "loop"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uid", 3) != 0)
        break;
      return AttributeCommonInfo::AT_Uuid;	 // "uuid"
    }
    break;
  case 6:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ranch", 5) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLControlFlowHint;	 // "branch"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "hader", 5) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLShader;	 // "shader"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nroll", 5) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLLoopHint;	 // "unroll"
    }
    break;
  case 7:	 // 1 string to match.
    if (memcmp(Name.data()+0, "flatten", 7) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLControlFlowHint;	 // "flatten"
  case 8:	 // 1 string to match.
    if (memcmp(Name.data()+0, "WaveSize", 8) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLWaveSize;	 // "WaveSize"
  case 10:	 // 1 string to match.
    if (memcmp(Name.data()+0, "numthreads", 10) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLNumThreads;	 // "numthreads"
  }
  } else if (AttributeCommonInfo::AS_CXX11 == Syntax) {
  switch (Name.size()) {
  default: break;
  case 6:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ssume", 5) != 0)
        break;
      return AttributeCommonInfo::AT_CXXAssume;	 // "assume"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ikely", 5) != 0)
        break;
      return AttributeCommonInfo::AT_Likely;	 // "likely"
    }
    break;
  case 8:	 // 5 strings to match.
    switch (Name[0]) {
    default: break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ar", 2) != 0)
          break;
        return AttributeCommonInfo::AT_MipsLongCall;	 // "gnu::far"
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ot", 2) != 0)
          break;
        return AttributeCommonInfo::AT_Hot;	 // "gnu::hot"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+6, "cs", 2) != 0)
          break;
        return AttributeCommonInfo::AT_Pcs;	 // "gnu::pcs"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oreturn", 7) != 0)
        break;
      return AttributeCommonInfo::AT_CXX11NoReturn;	 // "noreturn"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nlikely", 7) != 0)
        break;
      return AttributeCommonInfo::AT_Unlikely;	 // "unlikely"
    }
    break;
  case 9:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'g':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "old", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Cold;	 // "gnu::cold"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eaf", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Leaf;	 // "gnu::leaf"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ode", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Mode;	 // "gnu::mode"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ear", 3) != 0)
          break;
        return AttributeCommonInfo::AT_MipsShortCall;	 // "gnu::near"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ure", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Pure;	 // "gnu::pure"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+6, "sed", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Used;	 // "gnu::used"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eak", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Weak;	 // "gnu::weak"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "odiscard", 8) != 0)
        break;
      return AttributeCommonInfo::AT_WarnUnusedResult;	 // "nodiscard"
    }
    break;
  case 10:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eprecated", 9) != 0)
        break;
      return AttributeCommonInfo::AT_Deprecated;	 // "deprecated"
    case 'g':	 // 8 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 7 strings to match.
        if (memcmp(Name.data()+2, "u::", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+6, "lias", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Alias;	 // "gnu::alias"
        case 'c':	 // 2 strings to match.
          switch (Name[6]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+7, "ecl", 3) != 0)
              break;
            return AttributeCommonInfo::AT_CDecl;	 // "gnu::cdecl"
          case 'o':	 // 1 string to match.
            if (memcmp(Name.data()+7, "nst", 3) != 0)
              break;
            return AttributeCommonInfo::AT_Const;	 // "gnu::const"
          }
          break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+6, "rror", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Error;	 // "gnu::error"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+6, "func", 4) != 0)
            break;
          return AttributeCommonInfo::AT_IFunc;	 // "gnu::ifunc"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+6, "odel", 4) != 0)
            break;
          return AttributeCommonInfo::AT_CodeModel;	 // "gnu::model"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+6, "aked", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Naked;	 // "gnu::naked"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "l::Owner", 8) != 0)
          break;
        return AttributeCommonInfo::AT_Owner;	 // "gsl::Owner"
      }
      break;
    }
    break;
  case 11:	 // 12 strings to match.
    switch (Name[0]) {
    default: break;
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "allthrough", 10) != 0)
        break;
      return AttributeCommonInfo::AT_FallThrough;	 // "fallthrough"
    case 'g':	 // 10 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ommon", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Common;	 // "gnu::common"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ormat", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Format;	 // "gnu::format"
      case 'm':	 // 3 strings to match.
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "lloc", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Restrict;	 // "gnu::malloc"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ps16", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Mips16;	 // "gnu::mips16"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+7, "_abi", 4) != 0)
            break;
          return AttributeCommonInfo::AT_MSABI;	 // "gnu::ms_abi"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+6, "acked", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Packed;	 // "gnu::packed"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+6, "etain", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Retain;	 // "gnu::retain"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ignal", 5) != 0)
          break;
        return AttributeCommonInfo::AT_AVRSignal;	 // "gnu::signal"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arget", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Target;	 // "gnu::target"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nused", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Unused;	 // "gnu::unused"
      }
      break;
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mp::assume", 10) != 0)
        break;
      return AttributeCommonInfo::AT_OMPAssume;	 // "omp::assume"
    }
    break;
  case 12:	 // 18 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lang::guard", 11) != 0)
        break;
      return AttributeCommonInfo::AT_CFGuard;	 // "clang::guard"
    case 'g':	 // 15 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 14 strings to match.
        if (memcmp(Name.data()+2, "u::", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case '_':	 // 1 string to match.
          if (memcmp(Name.data()+6, "_const", 6) != 0)
            break;
          return AttributeCommonInfo::AT_Const;	 // "gnu::__const"
        case 'a':	 // 2 strings to match.
          switch (Name[6]) {
          default: break;
          case 'b':	 // 1 string to match.
            if (memcmp(Name.data()+7, "i_tag", 5) != 0)
              break;
            return AttributeCommonInfo::AT_AbiTag;	 // "gnu::abi_tag"
          case 'l':	 // 1 string to match.
            if (memcmp(Name.data()+7, "igned", 5) != 0)
              break;
            return AttributeCommonInfo::AT_Aligned;	 // "gnu::aligned"
          }
          break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+6, "leanup", 6) != 0)
            break;
          return AttributeCommonInfo::AT_Cleanup;	 // "gnu::cleanup"
        case 'f':	 // 1 string to match.
          if (memcmp(Name.data()+6, "latten", 6) != 0)
            break;
          return AttributeCommonInfo::AT_Flatten;	 // "gnu::flatten"
        case 'n':	 // 3 strings to match.
          if (Name[6] != 'o')
            break;
          switch (Name[7]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+8, "ebug", 4) != 0)
              break;
            return AttributeCommonInfo::AT_NoDebug;	 // "gnu::nodebug"
          case 'n':	 // 1 string to match.
            if (memcmp(Name.data()+8, "null", 4) != 0)
              break;
            return AttributeCommonInfo::AT_NonNull;	 // "gnu::nonnull"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+8, "hrow", 4) != 0)
              break;
            return AttributeCommonInfo::AT_NoThrow;	 // "gnu::nothrow"
          }
          break;
        case 'r':	 // 2 strings to match.
          if (memcmp(Name.data()+6, "eg", 2) != 0)
            break;
          switch (Name[8]) {
          default: break;
          case 'c':	 // 1 string to match.
            if (memcmp(Name.data()+9, "all", 3) != 0)
              break;
            return AttributeCommonInfo::AT_RegCall;	 // "gnu::regcall"
          case 'p':	 // 1 string to match.
            if (memcmp(Name.data()+9, "arm", 3) != 0)
              break;
            return AttributeCommonInfo::AT_Regparm;	 // "gnu::regparm"
          }
          break;
        case 's':	 // 2 strings to match.
          switch (Name[6]) {
          default: break;
          case 'e':	 // 1 string to match.
            if (memcmp(Name.data()+7, "ction", 5) != 0)
              break;
            return AttributeCommonInfo::AT_Section;	 // "gnu::section"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+7, "dcall", 5) != 0)
              break;
            return AttributeCommonInfo::AT_StdCall;	 // "gnu::stdcall"
          }
          break;
        case 'w':	 // 2 strings to match.
          switch (Name[6]) {
          default: break;
          case 'a':	 // 1 string to match.
            if (memcmp(Name.data()+7, "rning", 5) != 0)
              break;
            return AttributeCommonInfo::AT_Error;	 // "gnu::warning"
          case 'e':	 // 1 string to match.
            if (memcmp(Name.data()+7, "akref", 5) != 0)
              break;
            return AttributeCommonInfo::AT_WeakRef;	 // "gnu::weakref"
          }
          break;
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "l::Pointer", 10) != 0)
          break;
        return AttributeCommonInfo::AT_Pointer;	 // "gsl::Pointer"
      }
      break;
    case 'h':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lsl::is_rov", 11) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLROV;	 // "hlsl::is_rov"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "aybe_unused", 11) != 0)
        break;
      return AttributeCommonInfo::AT_Unused;	 // "maybe_unused"
    }
    break;
  case 13:	 // 14 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 4 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "sume", 4) != 0)
            break;
          return AttributeCommonInfo::AT_CXXAssume;	 // "clang::assume"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+9, "omic", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Atomic;	 // "clang::atomic"
        }
        break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+8, "locks", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Blocks;	 // "clang::blocks"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ascal", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Pascal;	 // "clang::pascal"
      }
      break;
    case 'g':	 // 9 strings to match.
      switch (Name[1]) {
      default: break;
      case 'n':	 // 8 strings to match.
        if (memcmp(Name.data()+2, "u::", 3) != 0)
          break;
        switch (Name[5]) {
        default: break;
        case 'f':	 // 1 string to match.
          if (memcmp(Name.data()+6, "astcall", 7) != 0)
            break;
          return AttributeCommonInfo::AT_FastCall;	 // "gnu::fastcall"
        case 'n':	 // 4 strings to match.
          if (Name[6] != 'o')
            break;
          switch (Name[7]) {
          default: break;
          case 'c':	 // 1 string to match.
            if (memcmp(Name.data()+8, "ommon", 5) != 0)
              break;
            return AttributeCommonInfo::AT_NoCommon;	 // "gnu::nocommon"
          case 'i':	 // 1 string to match.
            if (memcmp(Name.data()+8, "nline", 5) != 0)
              break;
            return AttributeCommonInfo::AT_NoInline;	 // "gnu::noinline"
          case 'm':	 // 1 string to match.
            if (memcmp(Name.data()+8, "ips16", 5) != 0)
              break;
            return AttributeCommonInfo::AT_NoMips16;	 // "gnu::nomips16"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+8, "eturn", 5) != 0)
              break;
            return AttributeCommonInfo::AT_NoReturn;	 // "gnu::noreturn"
          }
          break;
        case 's':	 // 2 strings to match.
          switch (Name[6]) {
          default: break;
          case 'e':	 // 1 string to match.
            if (memcmp(Name.data()+7, "ntinel", 6) != 0)
              break;
            return AttributeCommonInfo::AT_Sentinel;	 // "gnu::sentinel"
          case 'y':	 // 1 string to match.
            if (memcmp(Name.data()+7, "sv_abi", 6) != 0)
              break;
            return AttributeCommonInfo::AT_SysVABI;	 // "gnu::sysv_abi"
          }
          break;
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+6, "hiscall", 7) != 0)
            break;
          return AttributeCommonInfo::AT_ThisCall;	 // "gnu::thiscall"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "l::suppress", 11) != 0)
          break;
        return AttributeCommonInfo::AT_Suppress;	 // "gsl::suppress"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iscv::vls_cc", 12) != 0)
        break;
      return AttributeCommonInfo::AT_RISCVVLSCC;	 // "riscv::vls_cc"
    }
    break;
  case 14:	 // 15 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+8, "insize", 6) != 0)
          break;
        return AttributeCommonInfo::AT_MinSize;	 // "clang::minsize"
      case 'n':	 // 2 strings to match.
        if (Name[8] != 'o')
          break;
        switch (Name[9]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+10, "eref", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoDeref;	 // "clang::noderef"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+10, "erge", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoMerge;	 // "clang::nomerge"
        }
        break;
      case 'o':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_gc", 5) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCGC;	 // "clang::objc_gc"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "tnone", 5) != 0)
            break;
          return AttributeCommonInfo::AT_OptimizeNone;	 // "clang::optnone"
        }
        break;
      }
      break;
    case 'g':	 // 9 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'd':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "ll", 2) != 0)
          break;
        switch (Name[8]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+9, "xport", 5) != 0)
            break;
          return AttributeCommonInfo::AT_DLLExport;	 // "gnu::dllexport"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+9, "mport", 5) != 0)
            break;
          return AttributeCommonInfo::AT_DLLImport;	 // "gnu::dllimport"
        }
        break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nterrupt", 8) != 0)
          break;
        return AttributeCommonInfo::AT_Interrupt;	 // "gnu::interrupt"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ong_call", 8) != 0)
          break;
        return AttributeCommonInfo::AT_MipsLongCall;	 // "gnu::long_call"
      case 'm':	 // 3 strings to match.
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "y_alias", 7) != 0)
            break;
          return AttributeCommonInfo::AT_MayAlias;	 // "gnu::may_alias"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+7, "cromips", 7) != 0)
            break;
          return AttributeCommonInfo::AT_MicroMips;	 // "gnu::micromips"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+7, "_struct", 7) != 0)
            break;
          return AttributeCommonInfo::AT_MSStruct;	 // "gnu::ms_struct"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+6, "electany", 8) != 0)
          break;
        return AttributeCommonInfo::AT_SelectAny;	 // "gnu::selectany"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ls_model", 8) != 0)
          break;
        return AttributeCommonInfo::AT_TLSModel;	 // "gnu::tls_model"
      }
      break;
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "svc::noinline", 13) != 0)
        break;
      return AttributeCommonInfo::AT_NoInline;	 // "msvc::noinline"
    }
    break;
  case 15:	 // 22 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 12 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'N':	 // 1 string to match.
        if (memcmp(Name.data()+8, "SObject", 7) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCNSObject;	 // "clang::NSObject"
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nnotate", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Annotate;	 // "clang::annotate"
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+8, "locking", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Blocking;	 // "clang::blocking"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "allback", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Callback;	 // "clang::callback"
      case 'i':	 // 2 strings to match.
        if (Name[8] != 'b')
          break;
        switch (Name[9]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+10, "ction", 5) != 0)
            break;
          return AttributeCommonInfo::AT_IBAction;	 // "clang::ibaction"
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+10, "utlet", 5) != 0)
            break;
          return AttributeCommonInfo::AT_IBOutlet;	 // "clang::iboutlet"
        }
        break;
      case 'm':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case '6':	 // 1 string to match.
          if (memcmp(Name.data()+9, "8k_rtd", 6) != 0)
            break;
          return AttributeCommonInfo::AT_M68kRTD;	 // "clang::m68k_rtd"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+9, "sttail", 6) != 0)
            break;
          return AttributeCommonInfo::AT_MustTail;	 // "clang::musttail"
        }
        break;
      case 'n':	 // 2 strings to match.
        if (Name[8] != 'o')
          break;
        switch (Name[9]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+10, "scape", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoEscape;	 // "clang::noescape"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+10, "nline", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoInline;	 // "clang::noinline"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+9, "zed_by", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SizedBy;	 // "clang::sized_by"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ppress", 6) != 0)
            break;
          return AttributeCommonInfo::AT_Suppress;	 // "clang::suppress"
        }
        break;
      }
      break;
    case 'g':	 // 9 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 2 strings to match.
        switch (Name[6]) {
        default: break;
        case 'l':	 // 1 string to match.
          if (memcmp(Name.data()+7, "loc_size", 8) != 0)
            break;
          return AttributeCommonInfo::AT_AllocSize;	 // "gnu::alloc_size"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+7, "tificial", 8) != 0)
            break;
          return AttributeCommonInfo::AT_Artificial;	 // "gnu::artificial"
        }
        break;
      case 'd':	 // 2 strings to match.
        if (Name[6] != 'e')
          break;
        switch (Name[7]) {
        default: break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+8, "recated", 7) != 0)
            break;
          return AttributeCommonInfo::AT_Deprecated;	 // "gnu::deprecated"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+8, "tructor", 7) != 0)
            break;
          return AttributeCommonInfo::AT_Destructor;	 // "gnu::destructor"
        }
        break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ormat_arg", 9) != 0)
          break;
        return AttributeCommonInfo::AT_FormatArg;	 // "gnu::format_arg"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nu_inline", 9) != 0)
          break;
        return AttributeCommonInfo::AT_GNUInline;	 // "gnu::gnu_inline"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ocf_check", 9) != 0)
          break;
        return AttributeCommonInfo::AT_AnyX86NoCfCheck;	 // "gnu::nocf_check"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+6, "hort_call", 9) != 0)
          break;
        return AttributeCommonInfo::AT_MipsShortCall;	 // "gnu::short_call"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+6, "isibility", 9) != 0)
          break;
        return AttributeCommonInfo::AT_Visibility;	 // "gnu::visibility"
      }
      break;
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "svc::constexpr", 14) != 0)
        break;
      return AttributeCommonInfo::AT_MSConstexpr;	 // "msvc::constexpr"
    }
    break;
  case 16:	 // 13 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 4 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+8, "lag_enum", 8) != 0)
          break;
        return AttributeCommonInfo::AT_FlagEnum;	 // "clang::flag_enum"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ouwtable", 8) != 0)
          break;
        return AttributeCommonInfo::AT_NoUwtable;	 // "clang::nouwtable"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wiftcall", 8) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftCall;	 // "clang::swiftcall"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ecreturn", 8) != 0)
          break;
        return AttributeCommonInfo::AT_VecReturn;	 // "clang::vecreturn"
      }
      break;
    case 'g':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "lloc_align", 10) != 0)
          break;
        return AttributeCommonInfo::AT_AllocAlign;	 // "gnu::alloc_align"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "onstructor", 10) != 0)
          break;
        return AttributeCommonInfo::AT_Constructor;	 // "gnu::constructor"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "allthrough", 10) != 0)
          break;
        return AttributeCommonInfo::AT_FallThrough;	 // "gnu::fallthrough"
      case 'n':	 // 2 strings to match.
        if (Name[6] != 'o')
          break;
        switch (Name[7]) {
        default: break;
        case '_':	 // 1 string to match.
          if (memcmp(Name.data()+8, "sanitize", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitize;	 // "gnu::no_sanitize"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+8, "icromips", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoMicroMips;	 // "gnu::nomicromips"
        }
        break;
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ector_size", 10) != 0)
          break;
        return AttributeCommonInfo::AT_VectorSize;	 // "gnu::vector_size"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arn_unused", 10) != 0)
          break;
        return AttributeCommonInfo::AT_WarnUnused;	 // "gnu::warn_unused"
      }
      break;
    case 'h':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lsl::raw_buffer", 15) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLRawBuffer;	 // "hlsl::raw_buffer"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iscv::vector_cc", 15) != 0)
        break;
      return AttributeCommonInfo::AT_RISCVVectorCC;	 // "riscv::vector_cc"
    }
    break;
  case 17:	 // 11 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 10 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "llocating", 9) != 0)
          break;
        return AttributeCommonInfo::AT_Allocating;	 // "clang::allocating"
      case 'c':	 // 5 strings to match.
        switch (Name[8]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+9, "pability", 8) != 0)
            break;
          return AttributeCommonInfo::AT_Capability;	 // "clang::capability"
        case 'o':	 // 4 strings to match.
          switch (Name[9]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+10, "e_align", 7) != 0)
              break;
            return AttributeCommonInfo::AT_CodeAlign;	 // "clang::code_align"
          case 'n':	 // 2 strings to match.
            switch (Name[10]) {
            default: break;
            case 's':	 // 1 string to match.
              if (memcmp(Name.data()+11, "umable", 6) != 0)
                break;
              return AttributeCommonInfo::AT_Consumable;	 // "clang::consumable"
            case 'v':	 // 1 string to match.
              if (memcmp(Name.data()+11, "ergent", 6) != 0)
                break;
              return AttributeCommonInfo::AT_Convergent;	 // "clang::convergent"
            }
            break;
          case 'u':	 // 1 string to match.
            if (memcmp(Name.data()+10, "nted_by", 7) != 0)
              break;
            return AttributeCommonInfo::AT_CountedBy;	 // "clang::counted_by"
          }
          break;
        }
        break;
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+8, "o_", 2) != 0)
          break;
        switch (Name[10]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+11, "uiltin", 6) != 0)
            break;
          return AttributeCommonInfo::AT_NoBuiltin;	 // "clang::no_builtin"
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+11, "estroy", 6) != 0)
            break;
          return AttributeCommonInfo::AT_NoDestroy;	 // "clang::no_destroy"
        }
        break;
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+8, "se_handle", 9) != 0)
          break;
        return AttributeCommonInfo::AT_UseHandle;	 // "clang::use_handle"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ectorcall", 9) != 0)
          break;
        return AttributeCommonInfo::AT_VectorCall;	 // "clang::vectorcall"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "o_unique_address", 16) != 0)
        break;
      return AttributeCommonInfo::AT_NoUniqueAddress;	 // "no_unique_address"
    }
    break;
  case 18:	 // 26 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 22 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rries_dependency", 16) != 0)
          break;
        return AttributeCommonInfo::AT_CarriesDependency;	 // "carries_dependency"
      case 'l':	 // 21 strings to match.
        if (memcmp(Name.data()+2, "ang::", 5) != 0)
          break;
        switch (Name[7]) {
        default: break;
        case 'c':	 // 2 strings to match.
          switch (Name[8]) {
          default: break;
          case 'a':	 // 1 string to match.
            if (memcmp(Name.data()+9, "lled_once", 9) != 0)
              break;
            return AttributeCommonInfo::AT_CalledOnce;	 // "clang::called_once"
          case 'f':	 // 1 string to match.
            if (memcmp(Name.data()+9, "_consumed", 9) != 0)
              break;
            return AttributeCommonInfo::AT_CFConsumed;	 // "clang::cf_consumed"
          }
          break;
        case 'e':	 // 2 strings to match.
          switch (Name[8]) {
          default: break;
          case 'n':	 // 1 string to match.
            if (memcmp(Name.data()+9, "force_tcb", 9) != 0)
              break;
            return AttributeCommonInfo::AT_EnforceTCB;	 // "clang::enforce_tcb"
          case 'x':	 // 1 string to match.
            if (memcmp(Name.data()+9, "port_name", 9) != 0)
              break;
            return AttributeCommonInfo::AT_WebAssemblyExportName;	 // "clang::export_name"
          }
          break;
        case 'f':	 // 1 string to match.
          if (memcmp(Name.data()+8, "allthrough", 10) != 0)
            break;
          return AttributeCommonInfo::AT_FallThrough;	 // "clang::fallthrough"
        case 'g':	 // 1 string to match.
          if (memcmp(Name.data()+8, "uarded_var", 10) != 0)
            break;
          return AttributeCommonInfo::AT_GuardedVar;	 // "clang::guarded_var"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+8, "mport_name", 10) != 0)
            break;
          return AttributeCommonInfo::AT_WebAssemblyImportName;	 // "clang::import_name"
        case 'm':	 // 2 strings to match.
          if (Name[8] != 'a')
            break;
          switch (Name[9]) {
          default: break;
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+10, "rix_type", 8) != 0)
              break;
            return AttributeCommonInfo::AT_MatrixType;	 // "clang::matrix_type"
          case 'y':	 // 1 string to match.
            if (memcmp(Name.data()+10, "be_undef", 8) != 0)
              break;
            return AttributeCommonInfo::AT_MaybeUndef;	 // "clang::maybe_undef"
          }
          break;
        case 'n':	 // 4 strings to match.
          switch (Name[8]) {
          default: break;
          case 'o':	 // 3 strings to match.
            switch (Name[9]) {
            default: break;
            case '_':	 // 1 string to match.
              if (memcmp(Name.data()+10, "sanitize", 8) != 0)
                break;
              return AttributeCommonInfo::AT_NoSanitize;	 // "clang::no_sanitize"
            case 'd':	 // 1 string to match.
              if (memcmp(Name.data()+10, "uplicate", 8) != 0)
                break;
              return AttributeCommonInfo::AT_NoDuplicate;	 // "clang::noduplicate"
            case 'n':	 // 1 string to match.
              if (memcmp(Name.data()+10, "blocking", 8) != 0)
                break;
              return AttributeCommonInfo::AT_NonBlocking;	 // "clang::nonblocking"
            }
            break;
          case 's':	 // 1 string to match.
            if (memcmp(Name.data()+9, "_consumed", 9) != 0)
              break;
            return AttributeCommonInfo::AT_NSConsumed;	 // "clang::ns_consumed"
          }
          break;
        case 'o':	 // 3 strings to match.
          switch (Name[8]) {
          default: break;
          case 'b':	 // 2 strings to match.
            if (memcmp(Name.data()+9, "jc_", 3) != 0)
              break;
            switch (Name[12]) {
            default: break;
            case 'b':	 // 1 string to match.
              if (memcmp(Name.data()+13, "ridge", 5) != 0)
                break;
              return AttributeCommonInfo::AT_ObjCBridge;	 // "clang::objc_bridge"
            case 'd':	 // 1 string to match.
              if (memcmp(Name.data()+13, "irect", 5) != 0)
                break;
              return AttributeCommonInfo::AT_ObjCDirect;	 // "clang::objc_direct"
            }
            break;
          case 's':	 // 1 string to match.
            if (memcmp(Name.data()+9, "_consumed", 9) != 0)
              break;
            return AttributeCommonInfo::AT_OSConsumed;	 // "clang::os_consumed"
          }
          break;
        case 's':	 // 2 strings to match.
          switch (Name[8]) {
          default: break;
          case 'w':	 // 1 string to match.
            if (memcmp(Name.data()+9, "ift_async", 9) != 0)
              break;
            return AttributeCommonInfo::AT_SwiftAsync;	 // "clang::swift_async"
          case 'y':	 // 1 string to match.
            if (memcmp(Name.data()+9, "cl_kernel", 9) != 0)
              break;
            return AttributeCommonInfo::AT_SYCLKernel;	 // "clang::sycl_kernel"
          }
          break;
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+8, "rivial_abi", 10) != 0)
            break;
          return AttributeCommonInfo::AT_TrivialABI;	 // "clang::trivial_abi"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+8, "navailable", 10) != 0)
            break;
          return AttributeCommonInfo::AT_Unavailable;	 // "clang::unavailable"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+8, "eak_import", 10) != 0)
            break;
          return AttributeCommonInfo::AT_WeakImport;	 // "clang::weak_import"
        }
        break;
      }
      break;
    case 'g':	 // 4 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "lways_inline", 12) != 0)
          break;
        return AttributeCommonInfo::AT_AlwaysInline;	 // "gnu::always_inline"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nit_priority", 12) != 0)
          break;
        return AttributeCommonInfo::AT_InitPriority;	 // "gnu::init_priority"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eturns_twice", 12) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnsTwice;	 // "gnu::returns_twice"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arget_clones", 12) != 0)
          break;
        return AttributeCommonInfo::AT_TargetClones;	 // "gnu::target_clones"
      }
      break;
    }
    break;
  case 19:	 // 18 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 14 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "vailability", 11) != 0)
          break;
        return AttributeCommonInfo::AT_Availability;	 // "clang::availability"
      case 'b':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "f_fastcall", 10) != 0)
            break;
          return AttributeCommonInfo::AT_BPFFastCall;	 // "clang::bpf_fastcall"
        case 't':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "f_", 2) != 0)
            break;
          switch (Name[11]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+12, "ecl_tag", 7) != 0)
              break;
            return AttributeCommonInfo::AT_BTFDeclTag;	 // "clang::btf_decl_tag"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+12, "ype_tag", 7) != 0)
              break;
            return AttributeCommonInfo::AT_BTFTypeTag;	 // "clang::btf_type_tag"
          }
          break;
        }
        break;
      case 'c':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ro_wrapper", 10) != 0)
            break;
          return AttributeCommonInfo::AT_CoroWrapper;	 // "clang::coro_wrapper"
        case 'p':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "u_", 2) != 0)
            break;
          switch (Name[11]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+12, "ispatch", 7) != 0)
              break;
            return AttributeCommonInfo::AT_CPUDispatch;	 // "clang::cpu_dispatch"
          case 's':	 // 1 string to match.
            if (memcmp(Name.data()+12, "pecific", 7) != 0)
              break;
            return AttributeCommonInfo::AT_CPUSpecific;	 // "clang::cpu_specific"
          }
          break;
        }
        break;
      case 'n':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+9, "convergent", 10) != 0)
            break;
          return AttributeCommonInfo::AT_NoConvergent;	 // "clang::noconvergent"
        case 'v':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ptx_kernel", 10) != 0)
            break;
          return AttributeCommonInfo::AT_NVPTXKernel;	 // "clang::nvptx_kernel"
        }
        break;
      case 'o':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_boxable", 10) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCBoxable;	 // "clang::objc_boxable"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "encl_local", 10) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLLocalAddressSpace;	 // "clang::opencl_local"
        case 'v':	 // 1 string to match.
          if (memcmp(Name.data()+9, "erloadable", 10) != 0)
            break;
          return AttributeCommonInfo::AT_Overloadable;	 // "clang::overloadable"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "reserve_all", 11) != 0)
          break;
        return AttributeCommonInfo::AT_PreserveAll;	 // "clang::preserve_all"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "iscv_vls_cc", 11) != 0)
          break;
        return AttributeCommonInfo::AT_RISCVVLSCC;	 // "clang::riscv_vls_cc"
      }
      break;
    case 'g':	 // 4 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ssume_aligned", 13) != 0)
          break;
        return AttributeCommonInfo::AT_AssumeAligned;	 // "gnu::assume_aligned"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ormat_matches", 13) != 0)
          break;
        return AttributeCommonInfo::AT_FormatMatches;	 // "gnu::format_matches"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "o_split_stack", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoSplitStack;	 // "gnu::no_split_stack"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arget_version", 13) != 0)
          break;
        return AttributeCommonInfo::AT_TargetVersion;	 // "gnu::target_version"
      }
      break;
    }
    break;
  case 20:	 // 21 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 17 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 4 strings to match.
        switch (Name[8]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+9, "dress_space", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AddressSpace;	 // "clang::address_space"
        case 'l':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ways_inline", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AlwaysInline;	 // "clang::always_inline"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+9, "dgpu_kernel", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AMDGPUKernelCall;	 // "clang::amdgpu_kernel"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+9, "notate_type", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AnnotateType;	 // "clang::annotate_type"
        }
        break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+8, "uiltin_alias", 12) != 0)
          break;
        return AttributeCommonInfo::AT_BuiltinAlias;	 // "clang::builtin_alias"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "allable_when", 12) != 0)
          break;
        return AttributeCommonInfo::AT_CallableWhen;	 // "clang::callable_when"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+8, "mport_module", 12) != 0)
          break;
        return AttributeCommonInfo::AT_WebAssemblyImportModule;	 // "clang::import_module"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ifetimebound", 12) != 0)
          break;
        return AttributeCommonInfo::AT_LifetimeBound;	 // "clang::lifetimebound"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "onallocating", 12) != 0)
          break;
        return AttributeCommonInfo::AT_NonAllocating;	 // "clang::nonallocating"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "pencl_global", 12) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLGlobalAddressSpace;	 // "clang::opencl_global"
      case 'p':	 // 2 strings to match.
        if (memcmp(Name.data()+8, "reserve_", 8) != 0)
          break;
        switch (Name[16]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+17, "ost", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreserveMost;	 // "clang::preserve_most"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+17, "one", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreserveNone;	 // "clang::preserve_none"
        }
        break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "einitializes", 12) != 0)
          break;
        return AttributeCommonInfo::AT_Reinitializes;	 // "clang::reinitializes"
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+9, "t_typestate", 11) != 0)
            break;
          return AttributeCommonInfo::AT_SetTypestate;	 // "clang::set_typestate"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ift_context", 11) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftContext;	 // "clang::swift_context"
        }
        break;
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ninitialized", 12) != 0)
          break;
        return AttributeCommonInfo::AT_Uninitialized;	 // "clang::uninitialized"
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ray_log_args", 12) != 0)
          break;
        return AttributeCommonInfo::AT_XRayLogArgs;	 // "clang::xray_log_args"
      }
      break;
    case 'g':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "unction_return", 14) != 0)
          break;
        return AttributeCommonInfo::AT_FunctionReturnThunks;	 // "gnu::function_return"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eturns_nonnull", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnsNonNull;	 // "gnu::returns_nonnull"
      }
      break;
    case 'h':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "lsl::", 5) != 0)
        break;
      switch (Name[6]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+7, "ontained_type", 13) != 0)
          break;
        return AttributeCommonInfo::AT_HLSLContainedType;	 // "hlsl::contained_type"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+7, "esource_class", 13) != 0)
          break;
        return AttributeCommonInfo::AT_HLSLResourceClass;	 // "hlsl::resource_class"
      }
      break;
    }
    break;
  case 21:	 // 14 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 13 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+9, "quire_handle", 12) != 0)
            break;
          return AttributeCommonInfo::AT_AcquireHandle;	 // "clang::acquire_handle"
        case 'l':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ways_destroy", 12) != 0)
            break;
          return AttributeCommonInfo::AT_AlwaysDestroy;	 // "clang::always_destroy"
        }
        break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ntel_ocl_bicc", 13) != 0)
          break;
        return AttributeCommonInfo::AT_IntelOclBicc;	 // "clang::intel_ocl_bicc"
      case 'o':	 // 4 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "jc_", 3) != 0)
            break;
          switch (Name[12]) {
          default: break;
          case 'e':	 // 1 string to match.
            if (memcmp(Name.data()+13, "xception", 8) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCException;	 // "clang::objc_exception"
          case 'o':	 // 1 string to match.
            if (memcmp(Name.data()+13, "wnership", 8) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCOwnership;	 // "clang::objc_ownership"
          }
          break;
        case 'p':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "encl_", 5) != 0)
            break;
          switch (Name[14]) {
          default: break;
          case 'g':	 // 1 string to match.
            if (memcmp(Name.data()+15, "eneric", 6) != 0)
              break;
            return AttributeCommonInfo::AT_OpenCLGenericAddressSpace;	 // "clang::opencl_generic"
          case 'p':	 // 1 string to match.
            if (memcmp(Name.data()+15, "rivate", 6) != 0)
              break;
            return AttributeCommonInfo::AT_OpenCLPrivateAddressSpace;	 // "clang::opencl_private"
          }
          break;
        }
        break;
      case 'p':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'r':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "eferred_", 8) != 0)
            break;
          switch (Name[17]) {
          default: break;
          case 'n':	 // 1 string to match.
            if (memcmp(Name.data()+18, "ame", 3) != 0)
              break;
            return AttributeCommonInfo::AT_PreferredName;	 // "clang::preferred_name"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+18, "ype", 3) != 0)
              break;
            return AttributeCommonInfo::AT_PreferredType;	 // "clang::preferred_type"
          }
          break;
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+9, "_guarded_var", 12) != 0)
            break;
          return AttributeCommonInfo::AT_PtGuardedVar;	 // "clang::pt_guarded_var"
        }
        break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "elease_handle", 13) != 0)
          break;
        return AttributeCommonInfo::AT_ReleaseHandle;	 // "clang::release_handle"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wiftasynccall", 13) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftAsyncCall;	 // "clang::swiftasynccall"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+8, "est_typestate", 13) != 0)
          break;
        return AttributeCommonInfo::AT_TestTypestate;	 // "clang::test_typestate"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::randomize_layout", 20) != 0)
        break;
      return AttributeCommonInfo::AT_RandomizeLayout;	 // "gnu::randomize_layout"
    }
    break;
  case 22:	 // 17 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 16 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+9, "rch64_sve_pcs", 13) != 0)
            break;
          return AttributeCommonInfo::AT_AArch64SVEPcs;	 // "clang::aarch64_sve_pcs"
        case 'm':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "dgpu_num_", 9) != 0)
            break;
          switch (Name[18]) {
          default: break;
          case 's':	 // 1 string to match.
            if (memcmp(Name.data()+19, "gpr", 3) != 0)
              break;
            return AttributeCommonInfo::AT_AMDGPUNumSGPR;	 // "clang::amdgpu_num_sgpr"
          case 'v':	 // 1 string to match.
            if (memcmp(Name.data()+19, "gpr", 3) != 0)
              break;
            return AttributeCommonInfo::AT_AMDGPUNumVGPR;	 // "clang::amdgpu_num_vgpr"
          }
          break;
        }
        break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "xt_vector_type", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ExtVectorType;	 // "clang::ext_vector_type"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ot_tail_called", 14) != 0)
          break;
        return AttributeCommonInfo::AT_NotTailCalled;	 // "clang::not_tail_called"
      case 'o':	 // 5 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "jc_", 3) != 0)
            break;
          switch (Name[12]) {
          default: break;
          case 'c':	 // 1 string to match.
            if (memcmp(Name.data()+13, "lass_stub", 9) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCClassStub;	 // "clang::objc_class_stub"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+13, "oot_class", 9) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCRootClass;	 // "clang::objc_root_class"
          }
          break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "encl_constant", 13) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLConstantAddressSpace;	 // "clang::opencl_constant"
        case 'w':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "nership_", 8) != 0)
            break;
          switch (Name[17]) {
          default: break;
          case 'h':	 // 1 string to match.
            if (memcmp(Name.data()+18, "olds", 4) != 0)
              break;
            return AttributeCommonInfo::AT_Ownership;	 // "clang::ownership_holds"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+18, "akes", 4) != 0)
              break;
            return AttributeCommonInfo::AT_Ownership;	 // "clang::ownership_takes"
          }
          break;
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "aram_typestate", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ParamTypestate;	 // "clang::param_typestate"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "iscv_vector_cc", 14) != 0)
          break;
        return AttributeCommonInfo::AT_RISCVVectorCC;	 // "clang::riscv_vector_cc"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "coped_lockable", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ScopedLockable;	 // "clang::scoped_lockable"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ype_visibility", 14) != 0)
          break;
        return AttributeCommonInfo::AT_TypeVisibility;	 // "clang::type_visibility"
      case 'u':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+9, "lock_function", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ReleaseCapability;	 // "clang::unlock_function"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ing_if_exists", 13) != 0)
            break;
          return AttributeCommonInfo::AT_UsingIfExists;	 // "clang::using_if_exists"
        }
        break;
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::transparent_union", 21) != 0)
        break;
      return AttributeCommonInfo::AT_TransparentUnion;	 // "gnu::transparent_union"
    }
    break;
  case 23:	 // 16 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 12 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "oro_return_type", 15) != 0)
          break;
        return AttributeCommonInfo::AT_CoroReturnType;	 // "clang::coro_return_type"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nforce_tcb_leaf", 15) != 0)
          break;
        return AttributeCommonInfo::AT_EnforceTCBLeaf;	 // "clang::enforce_tcb_leaf"
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ybrid_patchable", 15) != 0)
          break;
        return AttributeCommonInfo::AT_HybridPatchable;	 // "clang::hybrid_patchable"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nternal_linkage", 15) != 0)
          break;
        return AttributeCommonInfo::AT_InternalLinkage;	 // "clang::internal_linkage"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+8, "in_vector_width", 15) != 0)
          break;
        return AttributeCommonInfo::AT_MinVectorWidth;	 // "clang::min_vector_width"
      case 'n':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+9, "on_vector_type", 14) != 0)
            break;
          return AttributeCommonInfo::AT_NeonVectorType;	 // "clang::neon_vector_type"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "_consumes_self", 14) != 0)
            break;
          return AttributeCommonInfo::AT_NSConsumesSelf;	 // "clang::ns_consumes_self"
        }
        break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "s_consumes_this", 15) != 0)
          break;
        return AttributeCommonInfo::AT_OSConsumesThis;	 // "clang::os_consumes_this"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ass_object_size", 15) != 0)
          break;
        return AttributeCommonInfo::AT_PassObjectSize;	 // "clang::pass_object_size"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "eturn_typestate", 15) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnTypestate;	 // "clang::return_typestate"
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+9, "zed_by_or_null", 14) != 0)
            break;
          return AttributeCommonInfo::AT_SizedByOrNull;	 // "clang::sized_by_or_null"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+9, "andalone_debug", 14) != 0)
            break;
          return AttributeCommonInfo::AT_StandaloneDebug;	 // "clang::standalone_debug"
        }
        break;
      }
      break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "o_s", 3) != 0)
          break;
        switch (Name[9]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+10, "nitize_thread", 13) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "gnu::no_sanitize_thread"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+10, "ack_protector", 13) != 0)
            break;
          return AttributeCommonInfo::AT_NoStackProtector;	 // "gnu::no_stack_protector"
        }
        break;
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arn_unused_result", 17) != 0)
          break;
        return AttributeCommonInfo::AT_WarnUnusedResult;	 // "gnu::warn_unused_result"
      }
      break;
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "svc::no_unique_address", 22) != 0)
        break;
      return AttributeCommonInfo::AT_NoUniqueAddress;	 // "msvc::no_unique_address"
    }
    break;
  case 24:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ssert_capability", 16) != 0)
          break;
        return AttributeCommonInfo::AT_AssertCapability;	 // "clang::assert_capability"
      case 'o':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_runtime_name", 15) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRuntimeName;	 // "clang::objc_runtime_name"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+9, "nership_returns", 15) != 0)
            break;
          return AttributeCommonInfo::AT_Ownership;	 // "clang::ownership_returns"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'h':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ared_capability", 15) != 0)
            break;
          return AttributeCommonInfo::AT_Capability;	 // "clang::shared_capability"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ift_async_error", 15) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftAsyncError;	 // "clang::swift_async_error"
        }
        break;
      }
      break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "o_", 2) != 0)
          break;
        switch (Name[8]) {
        default: break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+9, "andomize_layout", 15) != 0)
            break;
          return AttributeCommonInfo::AT_NoRandomizeLayout;	 // "gnu::no_randomize_layout"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "anitize_address", 15) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "gnu::no_sanitize_address"
        }
        break;
      case 'z':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ero_call_used_regs", 18) != 0)
          break;
        return AttributeCommonInfo::AT_ZeroCallUsedRegs;	 // "gnu::zero_call_used_regs"
      }
      break;
    }
    break;
  case 25:	 // 18 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'a':	 // 2 strings to match.
      switch (Name[8]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+9, "rch64_vector_pcs", 16) != 0)
          break;
        return AttributeCommonInfo::AT_AArch64VectorPcs;	 // "clang::aarch64_vector_pcs"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+9, "quire_capability", 16) != 0)
          break;
        return AttributeCommonInfo::AT_AcquireCapability;	 // "clang::acquire_capability"
      }
      break;
    case 'c':	 // 2 strings to match.
      if (Name[8] != 'o')
        break;
      switch (Name[9]) {
      default: break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+10, "o_lifetimebound", 15) != 0)
          break;
        return AttributeCommonInfo::AT_CoroLifetimeBound;	 // "clang::coro_lifetimebound"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+10, "nted_by_or_null", 15) != 0)
          break;
        return AttributeCommonInfo::AT_CountedByOrNull;	 // "clang::counted_by_or_null"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+8, "isable_tail_calls", 17) != 0)
        break;
      return AttributeCommonInfo::AT_DisableTailCalls;	 // "clang::disable_tail_calls"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+8, "num_extensibility", 17) != 0)
        break;
      return AttributeCommonInfo::AT_EnumExtensibility;	 // "clang::enum_extensibility"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+8, "boutletcollection", 17) != 0)
        break;
      return AttributeCommonInfo::AT_IBOutletCollection;	 // "clang::iboutletcollection"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+8, "ig_server_routine", 17) != 0)
        break;
      return AttributeCommonInfo::AT_MIGServerRoutine;	 // "clang::mig_server_routine"
    case 'n':	 // 3 strings to match.
      if (memcmp(Name.data()+8, "o_s", 3) != 0)
        break;
      switch (Name[11]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+12, "nitize_memory", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "clang::no_sanitize_memory"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+12, "ecializations", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoSpecializations;	 // "clang::no_specializations"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+12, "ack_protector", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoStackProtector;	 // "clang::no_stack_protector"
      }
      break;
    case 'o':	 // 3 strings to match.
      switch (Name[8]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+9, "jc_", 3) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+13, "ethod_family", 12) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCMethodFamily;	 // "clang::objc_method_family"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+13, "onlazy_class", 12) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCNonLazyClass;	 // "clang::objc_nonlazy_class"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+9, "encl_global_host", 16) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLGlobalHostAddressSpace;	 // "clang::opencl_global_host"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+8, "elease_capability", 17) != 0)
        break;
      return AttributeCommonInfo::AT_ReleaseCapability;	 // "clang::release_capability"
    case 's':	 // 2 strings to match.
      switch (Name[8]) {
      default: break;
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+9, "ift_error_result", 16) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftErrorResult;	 // "clang::swift_error_result"
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+9, "cl_special_class", 16) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLSpecialClass;	 // "clang::sycl_special_class"
      }
      break;
    case 'w':	 // 1 string to match.
      if (memcmp(Name.data()+8, "arn_unused_result", 17) != 0)
        break;
      return AttributeCommonInfo::AT_WarnUnusedResult;	 // "clang::warn_unused_result"
    }
    break;
  case 26:	 // 16 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+8, "mdgpu_waves_per_eu", 18) != 0)
        break;
      return AttributeCommonInfo::AT_AMDGPUWavesPerEU;	 // "clang::amdgpu_waves_per_eu"
    case 'c':	 // 4 strings to match.
      switch (Name[8]) {
      default: break;
      case 'f':	 // 3 strings to match.
        if (Name[9] != '_')
          break;
        switch (Name[10]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+11, "udited_transfer", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFAuditedTransfer;	 // "clang::cf_audited_transfer"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+11, "eturns_retained", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFReturnsRetained;	 // "clang::cf_returns_retained"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+11, "nknown_transfer", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFUnknownTransfer;	 // "clang::cf_unknown_transfer"
        }
        break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+9, "ro_await_elidable", 17) != 0)
          break;
        return AttributeCommonInfo::AT_CoroAwaitElidable;	 // "clang::coro_await_elidable"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+8, "iagnose_as_builtin", 18) != 0)
        break;
      return AttributeCommonInfo::AT_DiagnoseAsBuiltin;	 // "clang::diagnose_as_builtin"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+8, "ifetime_capture_by", 18) != 0)
        break;
      return AttributeCommonInfo::AT_LifetimeCaptureBy;	 // "clang::lifetime_capture_by"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+8, "s_returns_retained", 18) != 0)
        break;
      return AttributeCommonInfo::AT_NSReturnsRetained;	 // "clang::ns_returns_retained"
    case 'o':	 // 5 strings to match.
      switch (Name[8]) {
      default: break;
      case 'b':	 // 4 strings to match.
        if (memcmp(Name.data()+9, "jc_", 3) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+13, "ridge_", 6) != 0)
            break;
          switch (Name[19]) {
          default: break;
          case 'm':	 // 1 string to match.
            if (memcmp(Name.data()+20, "utable", 6) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridgeMutable;	 // "clang::objc_bridge_mutable"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+20, "elated", 6) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridgeRelated;	 // "clang::objc_bridge_related"
          }
          break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+13, "irect_members", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDirectMembers;	 // "clang::objc_direct_members"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+13, "equires_super", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRequiresSuper;	 // "clang::objc_requires_super"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+9, "_returns_retained", 17) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsRetained;	 // "clang::os_returns_retained"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+8, "equires_capability", 18) != 0)
        break;
      return AttributeCommonInfo::AT_RequiresCapability;	 // "clang::requires_capability"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+8, "wift_async_context", 18) != 0)
        break;
      return AttributeCommonInfo::AT_SwiftAsyncContext;	 // "clang::swift_async_context"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+8, "nsafe_buffer_usage", 18) != 0)
        break;
      return AttributeCommonInfo::AT_UnsafeBufferUsage;	 // "clang::unsafe_buffer_usage"
    }
    break;
  case 27:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "lspv_libclc_builtin", 19) != 0)
          break;
        return AttributeCommonInfo::AT_ClspvLibclcBuiltin;	 // "clang::clspv_libclc_builtin"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+8, "oader_uninitialized", 19) != 0)
          break;
        return AttributeCommonInfo::AT_LoaderUninitialized;	 // "clang::loader_uninitialized"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "eon_polyvector_type", 19) != 0)
          break;
        return AttributeCommonInfo::AT_NeonPolyVectorType;	 // "clang::neon_polyvector_type"
      case 'o':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_runtime_visible", 18) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRuntimeVisible;	 // "clang::objc_runtime_visible"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "encl_global_device", 18) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLGlobalDeviceAddressSpace;	 // "clang::opencl_global_device"
        }
        break;
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_instrument_function", 26) != 0)
        break;
      return AttributeCommonInfo::AT_NoInstrumentFunction;	 // "gnu::no_instrument_function"
    }
    break;
  case 28:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 8 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+8, "to_visibility_public", 20) != 0)
          break;
        return AttributeCommonInfo::AT_LTOVisibilityPublic;	 // "clang::lto_visibility_public"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "bjc_precise_lifetime", 20) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCPreciseLifetime;	 // "clang::objc_precise_lifetime"
      case 'p':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+9, "inter_with_type_tag", 19) != 0)
            break;
          return AttributeCommonInfo::AT_ArgumentWithTypeTag;	 // "clang::pointer_with_type_tag"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+9, "eserve_access_index", 19) != 0)
            break;
          return AttributeCommonInfo::AT_BPFPreserveAccessIndex;	 // "clang::preserve_access_index"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'h':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ared_locks_required", 19) != 0)
            break;
          return AttributeCommonInfo::AT_RequiresCapability;	 // "clang::shared_locks_required"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ift_indirect_result", 19) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftIndirectResult;	 // "clang::swift_indirect_result"
        }
        break;
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ype_tag_for_datatype", 20) != 0)
          break;
        return AttributeCommonInfo::AT_TypeTagForDatatype;	 // "clang::type_tag_for_datatype"
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ray_never_instrument", 20) != 0)
          break;
        return AttributeCommonInfo::AT_XRayInstrument;	 // "clang::xray_never_instrument"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::force_align_arg_pointer", 27) != 0)
        break;
      return AttributeCommonInfo::AT_X86ForceAlignArgPointer;	 // "gnu::force_align_arg_pointer"
    }
    break;
  case 29:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "rgument_with_type_tag", 21) != 0)
          break;
        return AttributeCommonInfo::AT_ArgumentWithTypeTag;	 // "clang::argument_with_type_tag"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "xternal_source_symbol", 21) != 0)
          break;
        return AttributeCommonInfo::AT_ExternalSourceSymbol;	 // "clang::external_source_symbol"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "bjc_independent_class", 21) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCIndependentClass;	 // "clang::objc_independent_class"
      case 'p':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+9, "eserve_static_offset", 20) != 0)
            break;
          return AttributeCommonInfo::AT_BPFPreserveStaticOffset;	 // "clang::preserve_static_offset"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+9, "rauth_vtable_pointer", 20) != 0)
            break;
          return AttributeCommonInfo::AT_VTablePointerAuthentication;	 // "clang::ptrauth_vtable_pointer"
        }
        break;
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ry_acquire_capability", 21) != 0)
          break;
        return AttributeCommonInfo::AT_TryAcquireCapability;	 // "clang::try_acquire_capability"
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ray_always_instrument", 21) != 0)
          break;
        return AttributeCommonInfo::AT_XRayInstrument;	 // "clang::xray_always_instrument"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::patchable_function_entry", 28) != 0)
        break;
      return AttributeCommonInfo::AT_PatchableFunctionEntry;	 // "gnu::patchable_function_entry"
    }
    break;
  case 30:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "f_returns_not_retained", 22) != 0)
          break;
        return AttributeCommonInfo::AT_CFReturnsNotRetained;	 // "clang::cf_returns_not_retained"
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+8, "s_returns_", 10) != 0)
          break;
        switch (Name[18]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+19, "utoreleased", 11) != 0)
            break;
          return AttributeCommonInfo::AT_NSReturnsAutoreleased;	 // "clang::ns_returns_autoreleased"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+19, "ot_retained", 11) != 0)
            break;
          return AttributeCommonInfo::AT_NSReturnsNotRetained;	 // "clang::ns_returns_not_retained"
        }
        break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "s_returns_not_retained", 22) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsNotRetained;	 // "clang::os_returns_not_retained"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ycl_kernel_entry_point", 22) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLKernelEntryPoint;	 // "clang::sycl_kernel_entry_point"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_caller_saved_registers", 29) != 0)
        break;
      return AttributeCommonInfo::AT_AnyX86NoCallerSavedRegisters;	 // "gnu::no_caller_saved_registers"
    }
    break;
  case 31:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ssert_shared_capability", 23) != 0)
          break;
        return AttributeCommonInfo::AT_AssertCapability;	 // "clang::assert_shared_capability"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "fi_canonical_jump_table", 23) != 0)
          break;
        return AttributeCommonInfo::AT_CFICanonicalJumpTable;	 // "clang::cfi_canonical_jump_table"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "xclusive_locks_required", 23) != 0)
          break;
        return AttributeCommonInfo::AT_RequiresCapability;	 // "clang::exclusive_locks_required"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "bjc_externally_retained", 23) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCExternallyRetained;	 // "clang::objc_externally_retained"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ass_dynamic_object_size", 23) != 0)
          break;
        return AttributeCommonInfo::AT_PassObjectSize;	 // "clang::pass_dynamic_object_size"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_address_safety_analysis", 30) != 0)
        break;
      return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "gnu::no_address_safety_analysis"
    }
    break;
  case 32:	 // 5 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+8, "_clang_arm_builtin_alias", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ArmBuiltinAlias;	 // "clang::__clang_arm_builtin_alias"
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+8, "cquire_shared_capability", 24) != 0)
        break;
      return AttributeCommonInfo::AT_AcquireCapability;	 // "clang::acquire_shared_capability"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+8, "o_thread_safety_analysis", 24) != 0)
        break;
      return AttributeCommonInfo::AT_NoThreadSafetyAnalysis;	 // "clang::no_thread_safety_analysis"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "bjc_non_runtime_protocol", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCNonRuntimeProtocol;	 // "clang::objc_non_runtime_protocol"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+8, "elease_shared_capability", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ReleaseCapability;	 // "clang::release_shared_capability"
    }
    break;
  case 33:	 // 7 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+8, "mdgpu_max_num_work_groups", 25) != 0)
        break;
      return AttributeCommonInfo::AT_AMDGPUMaxNumWorkGroups;	 // "clang::amdgpu_max_num_work_groups"
    case 'c':	 // 2 strings to match.
      if (Name[8] != 'o')
        break;
      switch (Name[9]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+10, "sumable_auto_cast_state", 23) != 0)
          break;
        return AttributeCommonInfo::AT_ConsumableAutoCast;	 // "clang::consumable_auto_cast_state"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+10, "o_disable_lifetimebound", 23) != 0)
          break;
        return AttributeCommonInfo::AT_CoroDisableLifetimeBound;	 // "clang::coro_disable_lifetimebound"
      }
      break;
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "bjc_returns_inner_pointer", 25) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCReturnsInnerPointer;	 // "clang::objc_returns_inner_pointer"
    case 'r':	 // 2 strings to match.
      if (Name[8] != 'e')
        break;
      switch (Name[9]) {
      default: break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+10, "ease_generic_capability", 23) != 0)
          break;
        return AttributeCommonInfo::AT_ReleaseCapability;	 // "clang::release_generic_capability"
      case 'q':	 // 1 string to match.
        if (memcmp(Name.data()+10, "uires_shared_capability", 23) != 0)
          break;
        return AttributeCommonInfo::AT_RequiresCapability;	 // "clang::requires_shared_capability"
      }
      break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+8, "peculative_load_hardening", 25) != 0)
        break;
      return AttributeCommonInfo::AT_SpeculativeLoadHardening;	 // "clang::speculative_load_hardening"
    }
    break;
  case 34:	 // 5 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+8, "mdgpu_flat_work_group_size", 26) != 0)
        break;
      return AttributeCommonInfo::AT_AMDGPUFlatWorkGroupSize;	 // "clang::amdgpu_flat_work_group_size"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+8, "nforce_read_only_placement", 26) != 0)
        break;
      return AttributeCommonInfo::AT_ReadOnlyPlacement;	 // "clang::enforce_read_only_placement"
    case 'o':	 // 3 strings to match.
      switch (Name[8]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+9, "jc_", 3) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+13, "esignated_initializer", 21) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDesignatedInitializer;	 // "clang::objc_designated_initializer"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+13, "ubclassing_restricted", 21) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCSubclassingRestricted;	 // "clang::objc_subclassing_restricted"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+9, "_returns_retained_on_zero", 25) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsRetainedOnZero;	 // "clang::os_returns_retained_on_zero"
      }
      break;
    }
    break;
  case 35:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "lang::co", 8) != 0)
        break;
      switch (Name[9]) {
      default: break;
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+10, "sumable_set_state_on_read", 25) != 0)
          break;
        return AttributeCommonInfo::AT_ConsumableSetOnRead;	 // "clang::consumable_set_state_on_read"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+10, "o_await_elidable_argument", 25) != 0)
          break;
        return AttributeCommonInfo::AT_CoroAwaitElidableArgument;	 // "clang::coro_await_elidable_argument"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_profile_instrument_function", 34) != 0)
        break;
      return AttributeCommonInfo::AT_NoProfileFunction;	 // "gnu::no_profile_instrument_function"
    }
    break;
  case 36:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+8, "o_speculative_load_hardening", 28) != 0)
        break;
      return AttributeCommonInfo::AT_NoSpeculativeLoadHardening;	 // "clang::no_speculative_load_hardening"
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+8, "ry_acquire_shared_capability", 28) != 0)
        break;
      return AttributeCommonInfo::AT_TryAcquireCapability;	 // "clang::try_acquire_shared_capability"
    }
    break;
  case 38:	 // 4 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+8, "oro_only_destroy_when_complete", 30) != 0)
        break;
      return AttributeCommonInfo::AT_CoroOnlyDestroyWhenComplete;	 // "clang::coro_only_destroy_when_complete"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "s_returns_retained_on_non_zero", 30) != 0)
        break;
      return AttributeCommonInfo::AT_OSReturnsRetainedOnNonZero;	 // "clang::os_returns_retained_on_non_zero"
    case 'r':	 // 2 strings to match.
      if (memcmp(Name.data()+8, "equire_", 7) != 0)
        break;
      switch (Name[15]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+16, "onstant_initialization", 22) != 0)
          break;
        return AttributeCommonInfo::AT_ConstInit;	 // "clang::require_constant_initialization"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+16, "xplicit_initialization", 22) != 0)
          break;
        return AttributeCommonInfo::AT_ExplicitInit;	 // "clang::require_explicit_initialization"
      }
      break;
    }
    break;
  case 40:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::disable_sanitizer_instrumentation", 40) != 0)
      break;
    return AttributeCommonInfo::AT_DisableSanitizerInstrumentation;	 // "clang::disable_sanitizer_instrumentation"
  case 41:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::objc_requires_property_definitions", 41) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCRequiresPropertyDefs;	 // "clang::objc_requires_property_definitions"
  case 42:	 // 3 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+8, "_clang_arm_mve_strict_polymorphism", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ArmMveStrictPolymorphism;	 // "clang::__clang_arm_mve_strict_polymorphism"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+8, "xclude_from_explicit_instantiation", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ExcludeFromExplicitInstantiation;	 // "clang::exclude_from_explicit_instantiation"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "bjc_arc_weak_reference_unavailable", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ArcWeakrefUnavailable;	 // "clang::objc_arc_weak_reference_unavailable"
    }
    break;
  case 44:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::available_only_in_default_eval_method", 44) != 0)
      break;
    return AttributeCommonInfo::AT_AvailableOnlyInDefaultEvalMethod;	 // "clang::available_only_in_default_eval_method"
  case 53:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::objc_protocol_requires_explicit_implementation", 53) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCExplicitProtocolImpl;	 // "clang::objc_protocol_requires_explicit_implementation"
  }
  } else if (AttributeCommonInfo::AS_C23 == Syntax) {
  switch (Name.size()) {
  default: break;
  case 8:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ar", 2) != 0)
          break;
        return AttributeCommonInfo::AT_MipsLongCall;	 // "gnu::far"
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ot", 2) != 0)
          break;
        return AttributeCommonInfo::AT_Hot;	 // "gnu::hot"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+6, "cs", 2) != 0)
          break;
        return AttributeCommonInfo::AT_Pcs;	 // "gnu::pcs"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "oreturn", 7) != 0)
        break;
      return AttributeCommonInfo::AT_CXX11NoReturn;	 // "noreturn"
    }
    break;
  case 9:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "Noreturn", 8) != 0)
        break;
      return AttributeCommonInfo::AT_CXX11NoReturn;	 // "_Noreturn"
    case 'g':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "old", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Cold;	 // "gnu::cold"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eaf", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Leaf;	 // "gnu::leaf"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ode", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Mode;	 // "gnu::mode"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ear", 3) != 0)
          break;
        return AttributeCommonInfo::AT_MipsShortCall;	 // "gnu::near"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ure", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Pure;	 // "gnu::pure"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+6, "sed", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Used;	 // "gnu::used"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eak", 3) != 0)
          break;
        return AttributeCommonInfo::AT_Weak;	 // "gnu::weak"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "odiscard", 8) != 0)
        break;
      return AttributeCommonInfo::AT_WarnUnusedResult;	 // "nodiscard"
    }
    break;
  case 10:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eprecated", 9) != 0)
        break;
      return AttributeCommonInfo::AT_Deprecated;	 // "deprecated"
    case 'g':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "lias", 4) != 0)
          break;
        return AttributeCommonInfo::AT_Alias;	 // "gnu::alias"
      case 'c':	 // 2 strings to match.
        switch (Name[6]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ecl", 3) != 0)
            break;
          return AttributeCommonInfo::AT_CDecl;	 // "gnu::cdecl"
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+7, "nst", 3) != 0)
            break;
          return AttributeCommonInfo::AT_Const;	 // "gnu::const"
        }
        break;
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+6, "rror", 4) != 0)
          break;
        return AttributeCommonInfo::AT_Error;	 // "gnu::error"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+6, "func", 4) != 0)
          break;
        return AttributeCommonInfo::AT_IFunc;	 // "gnu::ifunc"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+6, "odel", 4) != 0)
          break;
        return AttributeCommonInfo::AT_CodeModel;	 // "gnu::model"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "aked", 4) != 0)
          break;
        return AttributeCommonInfo::AT_Naked;	 // "gnu::naked"
      }
      break;
    }
    break;
  case 11:	 // 11 strings to match.
    switch (Name[0]) {
    default: break;
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "allthrough", 10) != 0)
        break;
      return AttributeCommonInfo::AT_FallThrough;	 // "fallthrough"
    case 'g':	 // 10 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ommon", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Common;	 // "gnu::common"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ormat", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Format;	 // "gnu::format"
      case 'm':	 // 3 strings to match.
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "lloc", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Restrict;	 // "gnu::malloc"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ps16", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Mips16;	 // "gnu::mips16"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+7, "_abi", 4) != 0)
            break;
          return AttributeCommonInfo::AT_MSABI;	 // "gnu::ms_abi"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+6, "acked", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Packed;	 // "gnu::packed"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+6, "etain", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Retain;	 // "gnu::retain"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ignal", 5) != 0)
          break;
        return AttributeCommonInfo::AT_AVRSignal;	 // "gnu::signal"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arget", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Target;	 // "gnu::target"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nused", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Unused;	 // "gnu::unused"
      }
      break;
    }
    break;
  case 12:	 // 15 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lang::guard", 11) != 0)
        break;
      return AttributeCommonInfo::AT_CFGuard;	 // "clang::guard"
    case 'g':	 // 13 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case '_':	 // 1 string to match.
        if (memcmp(Name.data()+6, "_const", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Const;	 // "gnu::__const"
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ligned", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Aligned;	 // "gnu::aligned"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "leanup", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Cleanup;	 // "gnu::cleanup"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "latten", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Flatten;	 // "gnu::flatten"
      case 'n':	 // 3 strings to match.
        if (Name[6] != 'o')
          break;
        switch (Name[7]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+8, "ebug", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoDebug;	 // "gnu::nodebug"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+8, "null", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NonNull;	 // "gnu::nonnull"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+8, "hrow", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoThrow;	 // "gnu::nothrow"
        }
        break;
      case 'r':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "eg", 2) != 0)
          break;
        switch (Name[8]) {
        default: break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+9, "all", 3) != 0)
            break;
          return AttributeCommonInfo::AT_RegCall;	 // "gnu::regcall"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "arm", 3) != 0)
            break;
          return AttributeCommonInfo::AT_Regparm;	 // "gnu::regparm"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[6]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ction", 5) != 0)
            break;
          return AttributeCommonInfo::AT_Section;	 // "gnu::section"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+7, "dcall", 5) != 0)
            break;
          return AttributeCommonInfo::AT_StdCall;	 // "gnu::stdcall"
        }
        break;
      case 'w':	 // 2 strings to match.
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "rning", 5) != 0)
            break;
          return AttributeCommonInfo::AT_Error;	 // "gnu::warning"
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+7, "akref", 5) != 0)
            break;
          return AttributeCommonInfo::AT_WeakRef;	 // "gnu::weakref"
        }
        break;
      }
      break;
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "aybe_unused", 11) != 0)
        break;
      return AttributeCommonInfo::AT_Unused;	 // "maybe_unused"
    }
    break;
  case 13:	 // 14 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "sume", 4) != 0)
            break;
          return AttributeCommonInfo::AT_CXXAssume;	 // "clang::assume"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+9, "omic", 4) != 0)
            break;
          return AttributeCommonInfo::AT_Atomic;	 // "clang::atomic"
        }
        break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+8, "locks", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Blocks;	 // "clang::blocks"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ikely", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Likely;	 // "clang::likely"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ascal", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Pascal;	 // "clang::pascal"
      }
      break;
    case 'g':	 // 8 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "astcall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_FastCall;	 // "gnu::fastcall"
      case 'n':	 // 4 strings to match.
        if (Name[6] != 'o')
          break;
        switch (Name[7]) {
        default: break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+8, "ommon", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoCommon;	 // "gnu::nocommon"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+8, "nline", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoInline;	 // "gnu::noinline"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+8, "ips16", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoMips16;	 // "gnu::nomips16"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+8, "eturn", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoReturn;	 // "gnu::noreturn"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[6]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+7, "ntinel", 6) != 0)
            break;
          return AttributeCommonInfo::AT_Sentinel;	 // "gnu::sentinel"
        case 'y':	 // 1 string to match.
          if (memcmp(Name.data()+7, "sv_abi", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SysVABI;	 // "gnu::sysv_abi"
        }
        break;
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "hiscall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_ThisCall;	 // "gnu::thiscall"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iscv::vls_cc", 12) != 0)
        break;
      return AttributeCommonInfo::AT_RISCVVLSCC;	 // "riscv::vls_cc"
    }
    break;
  case 14:	 // 15 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+8, "insize", 6) != 0)
          break;
        return AttributeCommonInfo::AT_MinSize;	 // "clang::minsize"
      case 'n':	 // 2 strings to match.
        if (Name[8] != 'o')
          break;
        switch (Name[9]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+10, "eref", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoDeref;	 // "clang::noderef"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+10, "erge", 4) != 0)
            break;
          return AttributeCommonInfo::AT_NoMerge;	 // "clang::nomerge"
        }
        break;
      case 'o':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_gc", 5) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCGC;	 // "clang::objc_gc"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "tnone", 5) != 0)
            break;
          return AttributeCommonInfo::AT_OptimizeNone;	 // "clang::optnone"
        }
        break;
      }
      break;
    case 'g':	 // 9 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'd':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "ll", 2) != 0)
          break;
        switch (Name[8]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+9, "xport", 5) != 0)
            break;
          return AttributeCommonInfo::AT_DLLExport;	 // "gnu::dllexport"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+9, "mport", 5) != 0)
            break;
          return AttributeCommonInfo::AT_DLLImport;	 // "gnu::dllimport"
        }
        break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nterrupt", 8) != 0)
          break;
        return AttributeCommonInfo::AT_Interrupt;	 // "gnu::interrupt"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ong_call", 8) != 0)
          break;
        return AttributeCommonInfo::AT_MipsLongCall;	 // "gnu::long_call"
      case 'm':	 // 3 strings to match.
        switch (Name[6]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+7, "y_alias", 7) != 0)
            break;
          return AttributeCommonInfo::AT_MayAlias;	 // "gnu::may_alias"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+7, "cromips", 7) != 0)
            break;
          return AttributeCommonInfo::AT_MicroMips;	 // "gnu::micromips"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+7, "_struct", 7) != 0)
            break;
          return AttributeCommonInfo::AT_MSStruct;	 // "gnu::ms_struct"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+6, "electany", 8) != 0)
          break;
        return AttributeCommonInfo::AT_SelectAny;	 // "gnu::selectany"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ls_model", 8) != 0)
          break;
        return AttributeCommonInfo::AT_TLSModel;	 // "gnu::tls_model"
      }
      break;
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+1, "svc::noinline", 13) != 0)
        break;
      return AttributeCommonInfo::AT_NoInline;	 // "msvc::noinline"
    }
    break;
  case 15:	 // 22 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 13 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'N':	 // 1 string to match.
        if (memcmp(Name.data()+8, "SObject", 7) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCNSObject;	 // "clang::NSObject"
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nnotate", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Annotate;	 // "clang::annotate"
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+8, "locking", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Blocking;	 // "clang::blocking"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "allback", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Callback;	 // "clang::callback"
      case 'i':	 // 2 strings to match.
        if (Name[8] != 'b')
          break;
        switch (Name[9]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+10, "ction", 5) != 0)
            break;
          return AttributeCommonInfo::AT_IBAction;	 // "clang::ibaction"
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+10, "utlet", 5) != 0)
            break;
          return AttributeCommonInfo::AT_IBOutlet;	 // "clang::iboutlet"
        }
        break;
      case 'm':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case '6':	 // 1 string to match.
          if (memcmp(Name.data()+9, "8k_rtd", 6) != 0)
            break;
          return AttributeCommonInfo::AT_M68kRTD;	 // "clang::m68k_rtd"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+9, "sttail", 6) != 0)
            break;
          return AttributeCommonInfo::AT_MustTail;	 // "clang::musttail"
        }
        break;
      case 'n':	 // 2 strings to match.
        if (Name[8] != 'o')
          break;
        switch (Name[9]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+10, "scape", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoEscape;	 // "clang::noescape"
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+10, "nline", 5) != 0)
            break;
          return AttributeCommonInfo::AT_NoInline;	 // "clang::noinline"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'i':	 // 1 string to match.
          if (memcmp(Name.data()+9, "zed_by", 6) != 0)
            break;
          return AttributeCommonInfo::AT_SizedBy;	 // "clang::sized_by"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ppress", 6) != 0)
            break;
          return AttributeCommonInfo::AT_Suppress;	 // "clang::suppress"
        }
        break;
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nlikely", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Unlikely;	 // "clang::unlikely"
      }
      break;
    case 'g':	 // 9 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 2 strings to match.
        switch (Name[6]) {
        default: break;
        case 'l':	 // 1 string to match.
          if (memcmp(Name.data()+7, "loc_size", 8) != 0)
            break;
          return AttributeCommonInfo::AT_AllocSize;	 // "gnu::alloc_size"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+7, "tificial", 8) != 0)
            break;
          return AttributeCommonInfo::AT_Artificial;	 // "gnu::artificial"
        }
        break;
      case 'd':	 // 2 strings to match.
        if (Name[6] != 'e')
          break;
        switch (Name[7]) {
        default: break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+8, "recated", 7) != 0)
            break;
          return AttributeCommonInfo::AT_Deprecated;	 // "gnu::deprecated"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+8, "tructor", 7) != 0)
            break;
          return AttributeCommonInfo::AT_Destructor;	 // "gnu::destructor"
        }
        break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ormat_arg", 9) != 0)
          break;
        return AttributeCommonInfo::AT_FormatArg;	 // "gnu::format_arg"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+6, "nu_inline", 9) != 0)
          break;
        return AttributeCommonInfo::AT_GNUInline;	 // "gnu::gnu_inline"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ocf_check", 9) != 0)
          break;
        return AttributeCommonInfo::AT_AnyX86NoCfCheck;	 // "gnu::nocf_check"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+6, "hort_call", 9) != 0)
          break;
        return AttributeCommonInfo::AT_MipsShortCall;	 // "gnu::short_call"
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+6, "isibility", 9) != 0)
          break;
        return AttributeCommonInfo::AT_Visibility;	 // "gnu::visibility"
      }
      break;
    }
    break;
  case 16:	 // 11 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+8, "lag_enum", 8) != 0)
          break;
        return AttributeCommonInfo::AT_FlagEnum;	 // "clang::flag_enum"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ouwtable", 8) != 0)
          break;
        return AttributeCommonInfo::AT_NoUwtable;	 // "clang::nouwtable"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wiftcall", 8) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftCall;	 // "clang::swiftcall"
      }
      break;
    case 'g':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "lloc_align", 10) != 0)
          break;
        return AttributeCommonInfo::AT_AllocAlign;	 // "gnu::alloc_align"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+6, "onstructor", 10) != 0)
          break;
        return AttributeCommonInfo::AT_Constructor;	 // "gnu::constructor"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "allthrough", 10) != 0)
          break;
        return AttributeCommonInfo::AT_FallThrough;	 // "gnu::fallthrough"
      case 'n':	 // 2 strings to match.
        if (Name[6] != 'o')
          break;
        switch (Name[7]) {
        default: break;
        case '_':	 // 1 string to match.
          if (memcmp(Name.data()+8, "sanitize", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitize;	 // "gnu::no_sanitize"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+8, "icromips", 8) != 0)
            break;
          return AttributeCommonInfo::AT_NoMicroMips;	 // "gnu::nomicromips"
        }
        break;
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ector_size", 10) != 0)
          break;
        return AttributeCommonInfo::AT_VectorSize;	 // "gnu::vector_size"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arn_unused", 10) != 0)
          break;
        return AttributeCommonInfo::AT_WarnUnused;	 // "gnu::warn_unused"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "iscv::vector_cc", 15) != 0)
        break;
      return AttributeCommonInfo::AT_RISCVVectorCC;	 // "riscv::vector_cc"
    }
    break;
  case 17:	 // 7 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+8, "llocating", 9) != 0)
        break;
      return AttributeCommonInfo::AT_Allocating;	 // "clang::allocating"
    case 'c':	 // 3 strings to match.
      if (Name[8] != 'o')
        break;
      switch (Name[9]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+10, "e_align", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CodeAlign;	 // "clang::code_align"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+10, "vergent", 7) != 0)
          break;
        return AttributeCommonInfo::AT_Convergent;	 // "clang::convergent"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+10, "nted_by", 7) != 0)
          break;
        return AttributeCommonInfo::AT_CountedBy;	 // "clang::counted_by"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+8, "o_builtin", 9) != 0)
        break;
      return AttributeCommonInfo::AT_NoBuiltin;	 // "clang::no_builtin"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+8, "se_handle", 9) != 0)
        break;
      return AttributeCommonInfo::AT_UseHandle;	 // "clang::use_handle"
    case 'v':	 // 1 string to match.
      if (memcmp(Name.data()+8, "ectorcall", 9) != 0)
        break;
      return AttributeCommonInfo::AT_VectorCall;	 // "clang::vectorcall"
    }
    break;
  case 18:	 // 21 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 18 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+9, "lled_once", 9) != 0)
            break;
          return AttributeCommonInfo::AT_CalledOnce;	 // "clang::called_once"
        case 'f':	 // 1 string to match.
          if (memcmp(Name.data()+9, "_consumed", 9) != 0)
            break;
          return AttributeCommonInfo::AT_CFConsumed;	 // "clang::cf_consumed"
        }
        break;
      case 'e':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+9, "force_tcb", 9) != 0)
            break;
          return AttributeCommonInfo::AT_EnforceTCB;	 // "clang::enforce_tcb"
        case 'x':	 // 1 string to match.
          if (memcmp(Name.data()+9, "port_name", 9) != 0)
            break;
          return AttributeCommonInfo::AT_WebAssemblyExportName;	 // "clang::export_name"
        }
        break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+8, "mport_name", 10) != 0)
          break;
        return AttributeCommonInfo::AT_WebAssemblyImportName;	 // "clang::import_name"
      case 'm':	 // 2 strings to match.
        if (Name[8] != 'a')
          break;
        switch (Name[9]) {
        default: break;
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+10, "rix_type", 8) != 0)
            break;
          return AttributeCommonInfo::AT_MatrixType;	 // "clang::matrix_type"
        case 'y':	 // 1 string to match.
          if (memcmp(Name.data()+10, "be_undef", 8) != 0)
            break;
          return AttributeCommonInfo::AT_MaybeUndef;	 // "clang::maybe_undef"
        }
        break;
      case 'n':	 // 4 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 3 strings to match.
          switch (Name[9]) {
          default: break;
          case '_':	 // 1 string to match.
            if (memcmp(Name.data()+10, "sanitize", 8) != 0)
              break;
            return AttributeCommonInfo::AT_NoSanitize;	 // "clang::no_sanitize"
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+10, "uplicate", 8) != 0)
              break;
            return AttributeCommonInfo::AT_NoDuplicate;	 // "clang::noduplicate"
          case 'n':	 // 1 string to match.
            if (memcmp(Name.data()+10, "blocking", 8) != 0)
              break;
            return AttributeCommonInfo::AT_NonBlocking;	 // "clang::nonblocking"
          }
          break;
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "_consumed", 9) != 0)
            break;
          return AttributeCommonInfo::AT_NSConsumed;	 // "clang::ns_consumed"
        }
        break;
      case 'o':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "jc_", 3) != 0)
            break;
          switch (Name[12]) {
          default: break;
          case 'b':	 // 1 string to match.
            if (memcmp(Name.data()+13, "ridge", 5) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridge;	 // "clang::objc_bridge"
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+13, "irect", 5) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCDirect;	 // "clang::objc_direct"
          }
          break;
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "_consumed", 9) != 0)
            break;
          return AttributeCommonInfo::AT_OSConsumed;	 // "clang::os_consumed"
        }
        break;
      case 's':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ift_async", 9) != 0)
            break;
          return AttributeCommonInfo::AT_SwiftAsync;	 // "clang::swift_async"
        case 'y':	 // 1 string to match.
          if (memcmp(Name.data()+9, "cl_kernel", 9) != 0)
            break;
          return AttributeCommonInfo::AT_SYCLKernel;	 // "clang::sycl_kernel"
        }
        break;
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+8, "navailable", 10) != 0)
          break;
        return AttributeCommonInfo::AT_Unavailable;	 // "clang::unavailable"
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+8, "eak_import", 10) != 0)
          break;
        return AttributeCommonInfo::AT_WeakImport;	 // "clang::weak_import"
      }
      break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "lways_inline", 12) != 0)
          break;
        return AttributeCommonInfo::AT_AlwaysInline;	 // "gnu::always_inline"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eturns_twice", 12) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnsTwice;	 // "gnu::returns_twice"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arget_clones", 12) != 0)
          break;
        return AttributeCommonInfo::AT_TargetClones;	 // "gnu::target_clones"
      }
      break;
    }
    break;
  case 19:	 // 18 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 14 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "vailability", 11) != 0)
          break;
        return AttributeCommonInfo::AT_Availability;	 // "clang::availability"
      case 'b':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "f_fastcall", 10) != 0)
            break;
          return AttributeCommonInfo::AT_BPFFastCall;	 // "clang::bpf_fastcall"
        case 't':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "f_", 2) != 0)
            break;
          switch (Name[11]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+12, "ecl_tag", 7) != 0)
              break;
            return AttributeCommonInfo::AT_BTFDeclTag;	 // "clang::btf_decl_tag"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+12, "ype_tag", 7) != 0)
              break;
            return AttributeCommonInfo::AT_BTFTypeTag;	 // "clang::btf_type_tag"
          }
          break;
        }
        break;
      case 'c':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ro_wrapper", 10) != 0)
            break;
          return AttributeCommonInfo::AT_CoroWrapper;	 // "clang::coro_wrapper"
        case 'p':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "u_", 2) != 0)
            break;
          switch (Name[11]) {
          default: break;
          case 'd':	 // 1 string to match.
            if (memcmp(Name.data()+12, "ispatch", 7) != 0)
              break;
            return AttributeCommonInfo::AT_CPUDispatch;	 // "clang::cpu_dispatch"
          case 's':	 // 1 string to match.
            if (memcmp(Name.data()+12, "pecific", 7) != 0)
              break;
            return AttributeCommonInfo::AT_CPUSpecific;	 // "clang::cpu_specific"
          }
          break;
        }
        break;
      case 'n':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+9, "convergent", 10) != 0)
            break;
          return AttributeCommonInfo::AT_NoConvergent;	 // "clang::noconvergent"
        case 'v':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ptx_kernel", 10) != 0)
            break;
          return AttributeCommonInfo::AT_NVPTXKernel;	 // "clang::nvptx_kernel"
        }
        break;
      case 'o':	 // 3 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_boxable", 10) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCBoxable;	 // "clang::objc_boxable"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "encl_local", 10) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLLocalAddressSpace;	 // "clang::opencl_local"
        case 'v':	 // 1 string to match.
          if (memcmp(Name.data()+9, "erloadable", 10) != 0)
            break;
          return AttributeCommonInfo::AT_Overloadable;	 // "clang::overloadable"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "reserve_all", 11) != 0)
          break;
        return AttributeCommonInfo::AT_PreserveAll;	 // "clang::preserve_all"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "iscv_vls_cc", 11) != 0)
          break;
        return AttributeCommonInfo::AT_RISCVVLSCC;	 // "clang::riscv_vls_cc"
      }
      break;
    case 'g':	 // 4 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ssume_aligned", 13) != 0)
          break;
        return AttributeCommonInfo::AT_AssumeAligned;	 // "gnu::assume_aligned"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ormat_matches", 13) != 0)
          break;
        return AttributeCommonInfo::AT_FormatMatches;	 // "gnu::format_matches"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+6, "o_split_stack", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoSplitStack;	 // "gnu::no_split_stack"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arget_version", 13) != 0)
          break;
        return AttributeCommonInfo::AT_TargetVersion;	 // "gnu::target_version"
      }
      break;
    }
    break;
  case 20:	 // 14 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 12 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 4 strings to match.
        switch (Name[8]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+9, "dress_space", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AddressSpace;	 // "clang::address_space"
        case 'l':	 // 1 string to match.
          if (memcmp(Name.data()+9, "ways_inline", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AlwaysInline;	 // "clang::always_inline"
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+9, "dgpu_kernel", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AMDGPUKernelCall;	 // "clang::amdgpu_kernel"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+9, "notate_type", 11) != 0)
            break;
          return AttributeCommonInfo::AT_AnnotateType;	 // "clang::annotate_type"
        }
        break;
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+8, "uiltin_alias", 12) != 0)
          break;
        return AttributeCommonInfo::AT_BuiltinAlias;	 // "clang::builtin_alias"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+8, "mport_module", 12) != 0)
          break;
        return AttributeCommonInfo::AT_WebAssemblyImportModule;	 // "clang::import_module"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "onallocating", 12) != 0)
          break;
        return AttributeCommonInfo::AT_NonAllocating;	 // "clang::nonallocating"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "pencl_global", 12) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLGlobalAddressSpace;	 // "clang::opencl_global"
      case 'p':	 // 2 strings to match.
        if (memcmp(Name.data()+8, "reserve_", 8) != 0)
          break;
        switch (Name[16]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+17, "ost", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreserveMost;	 // "clang::preserve_most"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+17, "one", 3) != 0)
            break;
          return AttributeCommonInfo::AT_PreserveNone;	 // "clang::preserve_none"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wift_context", 12) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftContext;	 // "clang::swift_context"
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ray_log_args", 12) != 0)
          break;
        return AttributeCommonInfo::AT_XRayLogArgs;	 // "clang::xray_log_args"
      }
      break;
    case 'g':	 // 2 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+6, "unction_return", 14) != 0)
          break;
        return AttributeCommonInfo::AT_FunctionReturnThunks;	 // "gnu::function_return"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+6, "eturns_nonnull", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ReturnsNonNull;	 // "gnu::returns_nonnull"
      }
      break;
    }
    break;
  case 21:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 8 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "cquire_handle", 13) != 0)
          break;
        return AttributeCommonInfo::AT_AcquireHandle;	 // "clang::acquire_handle"
      case 'o':	 // 4 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "jc_", 3) != 0)
            break;
          switch (Name[12]) {
          default: break;
          case 'e':	 // 1 string to match.
            if (memcmp(Name.data()+13, "xception", 8) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCException;	 // "clang::objc_exception"
          case 'o':	 // 1 string to match.
            if (memcmp(Name.data()+13, "wnership", 8) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCOwnership;	 // "clang::objc_ownership"
          }
          break;
        case 'p':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "encl_", 5) != 0)
            break;
          switch (Name[14]) {
          default: break;
          case 'g':	 // 1 string to match.
            if (memcmp(Name.data()+15, "eneric", 6) != 0)
              break;
            return AttributeCommonInfo::AT_OpenCLGenericAddressSpace;	 // "clang::opencl_generic"
          case 'p':	 // 1 string to match.
            if (memcmp(Name.data()+15, "rivate", 6) != 0)
              break;
            return AttributeCommonInfo::AT_OpenCLPrivateAddressSpace;	 // "clang::opencl_private"
          }
          break;
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "referred_type", 13) != 0)
          break;
        return AttributeCommonInfo::AT_PreferredType;	 // "clang::preferred_type"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "elease_handle", 13) != 0)
          break;
        return AttributeCommonInfo::AT_ReleaseHandle;	 // "clang::release_handle"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wiftasynccall", 13) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftAsyncCall;	 // "clang::swiftasynccall"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::randomize_layout", 20) != 0)
        break;
      return AttributeCommonInfo::AT_RandomizeLayout;	 // "gnu::randomize_layout"
    }
    break;
  case 22:	 // 11 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 10 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "arch64_sve_pcs", 14) != 0)
          break;
        return AttributeCommonInfo::AT_AArch64SVEPcs;	 // "clang::aarch64_sve_pcs"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "xt_vector_type", 14) != 0)
          break;
        return AttributeCommonInfo::AT_ExtVectorType;	 // "clang::ext_vector_type"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ot_tail_called", 14) != 0)
          break;
        return AttributeCommonInfo::AT_NotTailCalled;	 // "clang::not_tail_called"
      case 'o':	 // 5 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "jc_", 3) != 0)
            break;
          switch (Name[12]) {
          default: break;
          case 'c':	 // 1 string to match.
            if (memcmp(Name.data()+13, "lass_stub", 9) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCClassStub;	 // "clang::objc_class_stub"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+13, "oot_class", 9) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCRootClass;	 // "clang::objc_root_class"
          }
          break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "encl_constant", 13) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLConstantAddressSpace;	 // "clang::opencl_constant"
        case 'w':	 // 2 strings to match.
          if (memcmp(Name.data()+9, "nership_", 8) != 0)
            break;
          switch (Name[17]) {
          default: break;
          case 'h':	 // 1 string to match.
            if (memcmp(Name.data()+18, "olds", 4) != 0)
              break;
            return AttributeCommonInfo::AT_Ownership;	 // "clang::ownership_holds"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+18, "akes", 4) != 0)
              break;
            return AttributeCommonInfo::AT_Ownership;	 // "clang::ownership_takes"
          }
          break;
        }
        break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+8, "iscv_vector_cc", 14) != 0)
          break;
        return AttributeCommonInfo::AT_RISCVVectorCC;	 // "clang::riscv_vector_cc"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ype_visibility", 14) != 0)
          break;
        return AttributeCommonInfo::AT_TypeVisibility;	 // "clang::type_visibility"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::transparent_union", 21) != 0)
        break;
      return AttributeCommonInfo::AT_TransparentUnion;	 // "gnu::transparent_union"
    }
    break;
  case 23:	 // 13 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 10 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "oro_return_type", 15) != 0)
          break;
        return AttributeCommonInfo::AT_CoroReturnType;	 // "clang::coro_return_type"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nforce_tcb_leaf", 15) != 0)
          break;
        return AttributeCommonInfo::AT_EnforceTCBLeaf;	 // "clang::enforce_tcb_leaf"
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ybrid_patchable", 15) != 0)
          break;
        return AttributeCommonInfo::AT_HybridPatchable;	 // "clang::hybrid_patchable"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+8, "nternal_linkage", 15) != 0)
          break;
        return AttributeCommonInfo::AT_InternalLinkage;	 // "clang::internal_linkage"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+8, "in_vector_width", 15) != 0)
          break;
        return AttributeCommonInfo::AT_MinVectorWidth;	 // "clang::min_vector_width"
      case 'n':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'e':	 // 1 string to match.
          if (memcmp(Name.data()+9, "on_vector_type", 14) != 0)
            break;
          return AttributeCommonInfo::AT_NeonVectorType;	 // "clang::neon_vector_type"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "_consumes_self", 14) != 0)
            break;
          return AttributeCommonInfo::AT_NSConsumesSelf;	 // "clang::ns_consumes_self"
        }
        break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "s_consumes_this", 15) != 0)
          break;
        return AttributeCommonInfo::AT_OSConsumesThis;	 // "clang::os_consumes_this"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ass_object_size", 15) != 0)
          break;
        return AttributeCommonInfo::AT_PassObjectSize;	 // "clang::pass_object_size"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ized_by_or_null", 15) != 0)
          break;
        return AttributeCommonInfo::AT_SizedByOrNull;	 // "clang::sized_by_or_null"
      }
      break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "o_s", 3) != 0)
          break;
        switch (Name[9]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+10, "nitize_thread", 13) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "gnu::no_sanitize_thread"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+10, "ack_protector", 13) != 0)
            break;
          return AttributeCommonInfo::AT_NoStackProtector;	 // "gnu::no_stack_protector"
        }
        break;
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+6, "arn_unused_result", 17) != 0)
          break;
        return AttributeCommonInfo::AT_WarnUnusedResult;	 // "gnu::warn_unused_result"
      }
      break;
    }
    break;
  case 24:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'o':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_runtime_name", 15) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRuntimeName;	 // "clang::objc_runtime_name"
        case 'w':	 // 1 string to match.
          if (memcmp(Name.data()+9, "nership_returns", 15) != 0)
            break;
          return AttributeCommonInfo::AT_Ownership;	 // "clang::ownership_returns"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wift_async_error", 16) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftAsyncError;	 // "clang::swift_async_error"
      }
      break;
    case 'g':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "nu::", 4) != 0)
        break;
      switch (Name[5]) {
      default: break;
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+6, "o_", 2) != 0)
          break;
        switch (Name[8]) {
        default: break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+9, "andomize_layout", 15) != 0)
            break;
          return AttributeCommonInfo::AT_NoRandomizeLayout;	 // "gnu::no_randomize_layout"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+9, "anitize_address", 15) != 0)
            break;
          return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "gnu::no_sanitize_address"
        }
        break;
      case 'z':	 // 1 string to match.
        if (memcmp(Name.data()+6, "ero_call_used_regs", 18) != 0)
          break;
        return AttributeCommonInfo::AT_ZeroCallUsedRegs;	 // "gnu::zero_call_used_regs"
      }
      break;
    }
    break;
  case 25:	 // 14 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+8, "arch64_vector_pcs", 17) != 0)
        break;
      return AttributeCommonInfo::AT_AArch64VectorPcs;	 // "clang::aarch64_vector_pcs"
    case 'c':	 // 2 strings to match.
      if (Name[8] != 'o')
        break;
      switch (Name[9]) {
      default: break;
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+10, "o_lifetimebound", 15) != 0)
          break;
        return AttributeCommonInfo::AT_CoroLifetimeBound;	 // "clang::coro_lifetimebound"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+10, "nted_by_or_null", 15) != 0)
          break;
        return AttributeCommonInfo::AT_CountedByOrNull;	 // "clang::counted_by_or_null"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+8, "isable_tail_calls", 17) != 0)
        break;
      return AttributeCommonInfo::AT_DisableTailCalls;	 // "clang::disable_tail_calls"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+8, "num_extensibility", 17) != 0)
        break;
      return AttributeCommonInfo::AT_EnumExtensibility;	 // "clang::enum_extensibility"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+8, "boutletcollection", 17) != 0)
        break;
      return AttributeCommonInfo::AT_IBOutletCollection;	 // "clang::iboutletcollection"
    case 'm':	 // 1 string to match.
      if (memcmp(Name.data()+8, "ig_server_routine", 17) != 0)
        break;
      return AttributeCommonInfo::AT_MIGServerRoutine;	 // "clang::mig_server_routine"
    case 'n':	 // 2 strings to match.
      if (memcmp(Name.data()+8, "o_s", 3) != 0)
        break;
      switch (Name[11]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+12, "nitize_memory", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "clang::no_sanitize_memory"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+12, "ack_protector", 13) != 0)
          break;
        return AttributeCommonInfo::AT_NoStackProtector;	 // "clang::no_stack_protector"
      }
      break;
    case 'o':	 // 3 strings to match.
      switch (Name[8]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+9, "jc_", 3) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'm':	 // 1 string to match.
          if (memcmp(Name.data()+13, "ethod_family", 12) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCMethodFamily;	 // "clang::objc_method_family"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+13, "onlazy_class", 12) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCNonLazyClass;	 // "clang::objc_nonlazy_class"
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+9, "encl_global_host", 16) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLGlobalHostAddressSpace;	 // "clang::opencl_global_host"
      }
      break;
    case 's':	 // 2 strings to match.
      switch (Name[8]) {
      default: break;
      case 'w':	 // 1 string to match.
        if (memcmp(Name.data()+9, "ift_error_result", 16) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftErrorResult;	 // "clang::swift_error_result"
      case 'y':	 // 1 string to match.
        if (memcmp(Name.data()+9, "cl_special_class", 16) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLSpecialClass;	 // "clang::sycl_special_class"
      }
      break;
    }
    break;
  case 26:	 // 13 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'c':	 // 4 strings to match.
      switch (Name[8]) {
      default: break;
      case 'f':	 // 3 strings to match.
        if (Name[9] != '_')
          break;
        switch (Name[10]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+11, "udited_transfer", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFAuditedTransfer;	 // "clang::cf_audited_transfer"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+11, "eturns_retained", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFReturnsRetained;	 // "clang::cf_returns_retained"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+11, "nknown_transfer", 15) != 0)
            break;
          return AttributeCommonInfo::AT_CFUnknownTransfer;	 // "clang::cf_unknown_transfer"
        }
        break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+9, "ro_await_elidable", 17) != 0)
          break;
        return AttributeCommonInfo::AT_CoroAwaitElidable;	 // "clang::coro_await_elidable"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+8, "iagnose_as_builtin", 18) != 0)
        break;
      return AttributeCommonInfo::AT_DiagnoseAsBuiltin;	 // "clang::diagnose_as_builtin"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+8, "s_returns_retained", 18) != 0)
        break;
      return AttributeCommonInfo::AT_NSReturnsRetained;	 // "clang::ns_returns_retained"
    case 'o':	 // 5 strings to match.
      switch (Name[8]) {
      default: break;
      case 'b':	 // 4 strings to match.
        if (memcmp(Name.data()+9, "jc_", 3) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'b':	 // 2 strings to match.
          if (memcmp(Name.data()+13, "ridge_", 6) != 0)
            break;
          switch (Name[19]) {
          default: break;
          case 'm':	 // 1 string to match.
            if (memcmp(Name.data()+20, "utable", 6) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridgeMutable;	 // "clang::objc_bridge_mutable"
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+20, "elated", 6) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCBridgeRelated;	 // "clang::objc_bridge_related"
          }
          break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+13, "irect_members", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDirectMembers;	 // "clang::objc_direct_members"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+13, "equires_super", 13) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRequiresSuper;	 // "clang::objc_requires_super"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+9, "_returns_retained", 17) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsRetained;	 // "clang::os_returns_retained"
      }
      break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+8, "wift_async_context", 18) != 0)
        break;
      return AttributeCommonInfo::AT_SwiftAsyncContext;	 // "clang::swift_async_context"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+8, "nsafe_buffer_usage", 18) != 0)
        break;
      return AttributeCommonInfo::AT_UnsafeBufferUsage;	 // "clang::unsafe_buffer_usage"
    }
    break;
  case 27:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "lspv_libclc_builtin", 19) != 0)
          break;
        return AttributeCommonInfo::AT_ClspvLibclcBuiltin;	 // "clang::clspv_libclc_builtin"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+8, "oader_uninitialized", 19) != 0)
          break;
        return AttributeCommonInfo::AT_LoaderUninitialized;	 // "clang::loader_uninitialized"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+8, "eon_polyvector_type", 19) != 0)
          break;
        return AttributeCommonInfo::AT_NeonPolyVectorType;	 // "clang::neon_polyvector_type"
      case 'o':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'b':	 // 1 string to match.
          if (memcmp(Name.data()+9, "jc_runtime_visible", 18) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCRuntimeVisible;	 // "clang::objc_runtime_visible"
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+9, "encl_global_device", 18) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLGlobalDeviceAddressSpace;	 // "clang::opencl_global_device"
        }
        break;
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_instrument_function", 26) != 0)
        break;
      return AttributeCommonInfo::AT_NoInstrumentFunction;	 // "gnu::no_instrument_function"
    }
    break;
  case 28:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 7 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+8, "to_visibility_public", 20) != 0)
          break;
        return AttributeCommonInfo::AT_LTOVisibilityPublic;	 // "clang::lto_visibility_public"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "bjc_precise_lifetime", 20) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCPreciseLifetime;	 // "clang::objc_precise_lifetime"
      case 'p':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'o':	 // 1 string to match.
          if (memcmp(Name.data()+9, "inter_with_type_tag", 19) != 0)
            break;
          return AttributeCommonInfo::AT_ArgumentWithTypeTag;	 // "clang::pointer_with_type_tag"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+9, "eserve_access_index", 19) != 0)
            break;
          return AttributeCommonInfo::AT_BPFPreserveAccessIndex;	 // "clang::preserve_access_index"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "wift_indirect_result", 20) != 0)
          break;
        return AttributeCommonInfo::AT_SwiftIndirectResult;	 // "clang::swift_indirect_result"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ype_tag_for_datatype", 20) != 0)
          break;
        return AttributeCommonInfo::AT_TypeTagForDatatype;	 // "clang::type_tag_for_datatype"
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ray_never_instrument", 20) != 0)
          break;
        return AttributeCommonInfo::AT_XRayInstrument;	 // "clang::xray_never_instrument"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::force_align_arg_pointer", 27) != 0)
        break;
      return AttributeCommonInfo::AT_X86ForceAlignArgPointer;	 // "gnu::force_align_arg_pointer"
    }
    break;
  case 29:	 // 7 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 6 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+8, "rgument_with_type_tag", 21) != 0)
          break;
        return AttributeCommonInfo::AT_ArgumentWithTypeTag;	 // "clang::argument_with_type_tag"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+8, "xternal_source_symbol", 21) != 0)
          break;
        return AttributeCommonInfo::AT_ExternalSourceSymbol;	 // "clang::external_source_symbol"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "bjc_independent_class", 21) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCIndependentClass;	 // "clang::objc_independent_class"
      case 'p':	 // 2 strings to match.
        switch (Name[8]) {
        default: break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+9, "eserve_static_offset", 20) != 0)
            break;
          return AttributeCommonInfo::AT_BPFPreserveStaticOffset;	 // "clang::preserve_static_offset"
        case 't':	 // 1 string to match.
          if (memcmp(Name.data()+9, "rauth_vtable_pointer", 20) != 0)
            break;
          return AttributeCommonInfo::AT_VTablePointerAuthentication;	 // "clang::ptrauth_vtable_pointer"
        }
        break;
      case 'x':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ray_always_instrument", 21) != 0)
          break;
        return AttributeCommonInfo::AT_XRayInstrument;	 // "clang::xray_always_instrument"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::patchable_function_entry", 28) != 0)
        break;
      return AttributeCommonInfo::AT_PatchableFunctionEntry;	 // "gnu::patchable_function_entry"
    }
    break;
  case 30:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 5 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "f_returns_not_retained", 22) != 0)
          break;
        return AttributeCommonInfo::AT_CFReturnsNotRetained;	 // "clang::cf_returns_not_retained"
      case 'n':	 // 2 strings to match.
        if (memcmp(Name.data()+8, "s_returns_", 10) != 0)
          break;
        switch (Name[18]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+19, "utoreleased", 11) != 0)
            break;
          return AttributeCommonInfo::AT_NSReturnsAutoreleased;	 // "clang::ns_returns_autoreleased"
        case 'n':	 // 1 string to match.
          if (memcmp(Name.data()+19, "ot_retained", 11) != 0)
            break;
          return AttributeCommonInfo::AT_NSReturnsNotRetained;	 // "clang::ns_returns_not_retained"
        }
        break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "s_returns_not_retained", 22) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsNotRetained;	 // "clang::os_returns_not_retained"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ycl_kernel_entry_point", 22) != 0)
          break;
        return AttributeCommonInfo::AT_SYCLKernelEntryPoint;	 // "clang::sycl_kernel_entry_point"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_caller_saved_registers", 29) != 0)
        break;
      return AttributeCommonInfo::AT_AnyX86NoCallerSavedRegisters;	 // "gnu::no_caller_saved_registers"
    }
    break;
  case 31:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 3 strings to match.
      if (memcmp(Name.data()+1, "lang::", 6) != 0)
        break;
      switch (Name[7]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+8, "fi_canonical_jump_table", 23) != 0)
          break;
        return AttributeCommonInfo::AT_CFICanonicalJumpTable;	 // "clang::cfi_canonical_jump_table"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+8, "bjc_externally_retained", 23) != 0)
          break;
        return AttributeCommonInfo::AT_ObjCExternallyRetained;	 // "clang::objc_externally_retained"
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+8, "ass_dynamic_object_size", 23) != 0)
          break;
        return AttributeCommonInfo::AT_PassObjectSize;	 // "clang::pass_dynamic_object_size"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_address_safety_analysis", 30) != 0)
        break;
      return AttributeCommonInfo::AT_NoSanitizeSpecific;	 // "gnu::no_address_safety_analysis"
    }
    break;
  case 32:	 // 3 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+8, "_clang_arm_builtin_alias", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ArmBuiltinAlias;	 // "clang::__clang_arm_builtin_alias"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+8, "o_thread_safety_analysis", 24) != 0)
        break;
      return AttributeCommonInfo::AT_NoThreadSafetyAnalysis;	 // "clang::no_thread_safety_analysis"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "bjc_non_runtime_protocol", 24) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCNonRuntimeProtocol;	 // "clang::objc_non_runtime_protocol"
    }
    break;
  case 33:	 // 3 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+8, "oro_disable_lifetimebound", 25) != 0)
        break;
      return AttributeCommonInfo::AT_CoroDisableLifetimeBound;	 // "clang::coro_disable_lifetimebound"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "bjc_returns_inner_pointer", 25) != 0)
        break;
      return AttributeCommonInfo::AT_ObjCReturnsInnerPointer;	 // "clang::objc_returns_inner_pointer"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+8, "peculative_load_hardening", 25) != 0)
        break;
      return AttributeCommonInfo::AT_SpeculativeLoadHardening;	 // "clang::speculative_load_hardening"
    }
    break;
  case 34:	 // 4 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+8, "nforce_read_only_placement", 26) != 0)
        break;
      return AttributeCommonInfo::AT_ReadOnlyPlacement;	 // "clang::enforce_read_only_placement"
    case 'o':	 // 3 strings to match.
      switch (Name[8]) {
      default: break;
      case 'b':	 // 2 strings to match.
        if (memcmp(Name.data()+9, "jc_", 3) != 0)
          break;
        switch (Name[12]) {
        default: break;
        case 'd':	 // 1 string to match.
          if (memcmp(Name.data()+13, "esignated_initializer", 21) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCDesignatedInitializer;	 // "clang::objc_designated_initializer"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+13, "ubclassing_restricted", 21) != 0)
            break;
          return AttributeCommonInfo::AT_ObjCSubclassingRestricted;	 // "clang::objc_subclassing_restricted"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+9, "_returns_retained_on_zero", 25) != 0)
          break;
        return AttributeCommonInfo::AT_OSReturnsRetainedOnZero;	 // "clang::os_returns_retained_on_zero"
      }
      break;
    }
    break;
  case 35:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lang::coro_await_elidable_argument", 34) != 0)
        break;
      return AttributeCommonInfo::AT_CoroAwaitElidableArgument;	 // "clang::coro_await_elidable_argument"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nu::no_profile_instrument_function", 34) != 0)
        break;
      return AttributeCommonInfo::AT_NoProfileFunction;	 // "gnu::no_profile_instrument_function"
    }
    break;
  case 36:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::no_speculative_load_hardening", 36) != 0)
      break;
    return AttributeCommonInfo::AT_NoSpeculativeLoadHardening;	 // "clang::no_speculative_load_hardening"
  case 38:	 // 3 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+8, "oro_only_destroy_when_complete", 30) != 0)
        break;
      return AttributeCommonInfo::AT_CoroOnlyDestroyWhenComplete;	 // "clang::coro_only_destroy_when_complete"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "s_returns_retained_on_non_zero", 30) != 0)
        break;
      return AttributeCommonInfo::AT_OSReturnsRetainedOnNonZero;	 // "clang::os_returns_retained_on_non_zero"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+8, "equire_explicit_initialization", 30) != 0)
        break;
      return AttributeCommonInfo::AT_ExplicitInit;	 // "clang::require_explicit_initialization"
    }
    break;
  case 40:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::disable_sanitizer_instrumentation", 40) != 0)
      break;
    return AttributeCommonInfo::AT_DisableSanitizerInstrumentation;	 // "clang::disable_sanitizer_instrumentation"
  case 41:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::objc_requires_property_definitions", 41) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCRequiresPropertyDefs;	 // "clang::objc_requires_property_definitions"
  case 42:	 // 3 strings to match.
    if (memcmp(Name.data()+0, "clang::", 7) != 0)
      break;
    switch (Name[7]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+8, "_clang_arm_mve_strict_polymorphism", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ArmMveStrictPolymorphism;	 // "clang::__clang_arm_mve_strict_polymorphism"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+8, "xclude_from_explicit_instantiation", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ExcludeFromExplicitInstantiation;	 // "clang::exclude_from_explicit_instantiation"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+8, "bjc_arc_weak_reference_unavailable", 34) != 0)
        break;
      return AttributeCommonInfo::AT_ArcWeakrefUnavailable;	 // "clang::objc_arc_weak_reference_unavailable"
    }
    break;
  case 44:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::available_only_in_default_eval_method", 44) != 0)
      break;
    return AttributeCommonInfo::AT_AvailableOnlyInDefaultEvalMethod;	 // "clang::available_only_in_default_eval_method"
  case 53:	 // 1 string to match.
    if (memcmp(Name.data()+0, "clang::objc_protocol_requires_explicit_implementation", 53) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCExplicitProtocolImpl;	 // "clang::objc_protocol_requires_explicit_implementation"
  }
  } else if (AttributeCommonInfo::AS_Keyword == Syntax || AttributeCommonInfo::AS_ContextSensitiveKeyword == Syntax) {
  switch (Name.size()) {
  default: break;
  case 2:	 // 1 string to match.
    if (memcmp(Name.data()+0, "in", 2) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLParamModifier;	 // "in"
  case 3:	 // 1 string to match.
    if (memcmp(Name.data()+0, "out", 3) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLParamModifier;	 // "out"
  case 5:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 1 string to match.
      if (memcmp(Name.data()+1, "_w64", 4) != 0)
        break;
      return AttributeCommonInfo::IgnoredAttribute;	 // "__w64"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nout", 4) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLParamModifier;	 // "inout"
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ocal", 4) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLLocalAddressSpace;	 // "local"
    }
    break;
  case 6:	 // 5 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case '_':	 // 2 strings to match.
        switch (Name[2]) {
        default: break;
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+3, "ptr", 3) != 0)
            break;
          return AttributeCommonInfo::AT_SPtr;	 // "__sptr"
        case 'u':	 // 1 string to match.
          if (memcmp(Name.data()+3, "ptr", 3) != 0)
            break;
          return AttributeCommonInfo::AT_UPtr;	 // "__uptr"
        }
        break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "decl", 4) != 0)
          break;
        return AttributeCommonInfo::AT_CDecl;	 // "_cdecl"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lobal", 5) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLGlobalAddressSpace;	 // "global"
    case 'k':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ernel", 5) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLKernel;	 // "kernel"
    }
    break;
  case 7:	 // 8 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 5 strings to match.
      switch (Name[1]) {
      default: break;
      case '_':	 // 4 strings to match.
        switch (Name[2]) {
        default: break;
        case 'c':	 // 1 string to match.
          if (memcmp(Name.data()+3, "decl", 4) != 0)
            break;
          return AttributeCommonInfo::AT_CDecl;	 // "__cdecl"
        case 'l':	 // 1 string to match.
          if (memcmp(Name.data()+3, "ocal", 4) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLLocalAddressSpace;	 // "__local"
        case 'p':	 // 2 strings to match.
          if (memcmp(Name.data()+3, "tr", 2) != 0)
            break;
          switch (Name[5]) {
          default: break;
          case '3':	 // 1 string to match.
            if (Name[6] != '2')
              break;
            return AttributeCommonInfo::AT_Ptr32;	 // "__ptr32"
          case '6':	 // 1 string to match.
            if (Name[6] != '4')
              break;
            return AttributeCommonInfo::AT_Ptr64;	 // "__ptr64"
          }
          break;
        }
        break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ascal", 5) != 0)
          break;
        return AttributeCommonInfo::AT_Pascal;	 // "_pascal"
      }
      break;
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lignas", 6) != 0)
        break;
      return AttributeCommonInfo::AT_Aligned;	 // "alignas"
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eneric", 6) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLGenericAddressSpace;	 // "generic"
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rivate", 6) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLPrivateAddressSpace;	 // "private"
    }
    break;
  case 8:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 8 strings to match.
      switch (Name[1]) {
      default: break;
      case 'A':	 // 1 string to match.
        if (memcmp(Name.data()+2, "lignas", 6) != 0)
          break;
        return AttributeCommonInfo::AT_Aligned;	 // "_Alignas"
      case 'N':	 // 1 string to match.
        if (memcmp(Name.data()+2, "onnull", 6) != 0)
          break;
        return AttributeCommonInfo::AT_TypeNonNull;	 // "_Nonnull"
      case '_':	 // 5 strings to match.
        switch (Name[2]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+3, "rm_in", 5) != 0)
            break;
          return AttributeCommonInfo::AT_ArmIn;	 // "__arm_in"
        case 'g':	 // 1 string to match.
          if (memcmp(Name.data()+3, "lobal", 5) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLGlobalAddressSpace;	 // "__global"
        case 'k':	 // 2 strings to match.
          switch (Name[3]) {
          default: break;
          case 'e':	 // 1 string to match.
            if (memcmp(Name.data()+4, "rnel", 4) != 0)
              break;
            return AttributeCommonInfo::AT_OpenCLKernel;	 // "__kernel"
          case 'i':	 // 1 string to match.
            if (memcmp(Name.data()+4, "ndof", 4) != 0)
              break;
            return AttributeCommonInfo::AT_ObjCKindOf;	 // "__kindof"
          }
          break;
        case 'p':	 // 1 string to match.
          if (memcmp(Name.data()+3, "ascal", 5) != 0)
            break;
          return AttributeCommonInfo::AT_Pascal;	 // "__pascal"
        }
        break;
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "tdcall", 6) != 0)
          break;
        return AttributeCommonInfo::AT_StdCall;	 // "_stdcall"
      }
      break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "onstant", 7) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLConstantAddressSpace;	 // "constant"
    }
    break;
  case 9:	 // 13 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 11 strings to match.
      switch (Name[1]) {
      default: break;
      case 'N':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ullable", 7) != 0)
          break;
        return AttributeCommonInfo::AT_TypeNullable;	 // "_Nullable"
      case '_':	 // 8 strings to match.
        switch (Name[2]) {
        default: break;
        case 'a':	 // 2 strings to match.
          if (memcmp(Name.data()+3, "rm_", 3) != 0)
            break;
          switch (Name[6]) {
          default: break;
          case 'n':	 // 1 string to match.
            if (memcmp(Name.data()+7, "ew", 2) != 0)
              break;
            return AttributeCommonInfo::AT_ArmNew;	 // "__arm_new"
          case 'o':	 // 1 string to match.
            if (memcmp(Name.data()+7, "ut", 2) != 0)
              break;
            return AttributeCommonInfo::AT_ArmOut;	 // "__arm_out"
          }
          break;
        case 'f':	 // 1 string to match.
          if (memcmp(Name.data()+3, "uncref", 6) != 0)
            break;
          return AttributeCommonInfo::AT_WebAssemblyFuncref;	 // "__funcref"
        case 'g':	 // 1 string to match.
          if (memcmp(Name.data()+3, "eneric", 6) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLGenericAddressSpace;	 // "__generic"
        case 'p':	 // 2 strings to match.
          switch (Name[3]) {
          default: break;
          case 'r':	 // 1 string to match.
            if (memcmp(Name.data()+4, "ivate", 5) != 0)
              break;
            return AttributeCommonInfo::AT_OpenCLPrivateAddressSpace;	 // "__private"
          case 't':	 // 1 string to match.
            if (memcmp(Name.data()+4, "rauth", 5) != 0)
              break;
            return AttributeCommonInfo::AT_PointerAuth;	 // "__ptrauth"
          }
          break;
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+3, "egcall", 6) != 0)
            break;
          return AttributeCommonInfo::AT_RegCall;	 // "__regcall"
        case 's':	 // 1 string to match.
          if (memcmp(Name.data()+3, "tdcall", 6) != 0)
            break;
          return AttributeCommonInfo::AT_StdCall;	 // "__stdcall"
        }
        break;
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+2, "astcall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_FastCall;	 // "_fastcall"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "hiscall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_ThisCall;	 // "_thiscall"
      }
      break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "onstinit", 8) != 0)
        break;
      return AttributeCommonInfo::AT_ConstInit;	 // "constinit"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ead_only", 8) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLAccess;	 // "read_only"
    }
    break;
  case 10:	 // 5 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 3 strings to match.
      if (Name[1] != '_')
        break;
      switch (Name[2]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+3, "onstant", 7) != 0)
          break;
        return AttributeCommonInfo::AT_OpenCLConstantAddressSpace;	 // "__constant"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+3, "astcall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_FastCall;	 // "__fastcall"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+3, "hiscall", 7) != 0)
          break;
        return AttributeCommonInfo::AT_ThisCall;	 // "__thiscall"
      }
      break;
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ead_write", 9) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLAccess;	 // "read_write"
    case 'w':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rite_only", 9) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLAccess;	 // "write_only"
    }
    break;
  case 11:	 // 4 strings to match.
    switch (Name[0]) {
    default: break;
    case '_':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case '_':	 // 2 strings to match.
        switch (Name[2]) {
        default: break;
        case 'a':	 // 1 string to match.
          if (memcmp(Name.data()+3, "rm_inout", 8) != 0)
            break;
          return AttributeCommonInfo::AT_ArmInOut;	 // "__arm_inout"
        case 'r':	 // 1 string to match.
          if (memcmp(Name.data()+3, "ead_only", 8) != 0)
            break;
          return AttributeCommonInfo::AT_OpenCLAccess;	 // "__read_only"
        }
        break;
      case 'v':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ectorcall", 9) != 0)
          break;
        return AttributeCommonInfo::AT_VectorCall;	 // "_vectorcall"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "roupshared", 10) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLGroupSharedAddressSpace;	 // "groupshared"
    }
    break;
  case 12:	 // 4 strings to match.
    if (memcmp(Name.data()+0, "__", 2) != 0)
      break;
    switch (Name[2]) {
    default: break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+3, "oinline__", 9) != 0)
        break;
      return AttributeCommonInfo::AT_NoInline;	 // "__noinline__"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+3, "ead_write", 9) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLAccess;	 // "__read_write"
    case 'v':	 // 1 string to match.
      if (memcmp(Name.data()+3, "ectorcall", 9) != 0)
        break;
      return AttributeCommonInfo::AT_VectorCall;	 // "__vectorcall"
    case 'w':	 // 1 string to match.
      if (memcmp(Name.data()+3, "rite_only", 9) != 0)
        break;
      return AttributeCommonInfo::AT_OpenCLAccess;	 // "__write_only"
    }
    break;
  case 13:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__forceinline", 13) != 0)
      break;
    return AttributeCommonInfo::AT_AlwaysInline;	 // "__forceinline"
  case 14:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__arm_agnostic", 14) != 0)
      break;
    return AttributeCommonInfo::AT_ArmAgnostic;	 // "__arm_agnostic"
  case 15:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "__arm_", 6) != 0)
      break;
    switch (Name[6]) {
    default: break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+7, "reserves", 8) != 0)
        break;
      return AttributeCommonInfo::AT_ArmPreserves;	 // "__arm_preserves"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+7, "treaming", 8) != 0)
        break;
      return AttributeCommonInfo::AT_ArmStreaming;	 // "__arm_streaming"
    }
    break;
  case 16:	 // 1 string to match.
    if (memcmp(Name.data()+0, "_Nullable_result", 16) != 0)
      break;
    return AttributeCommonInfo::AT_TypeNullableResult;	 // "_Nullable_result"
  case 17:	 // 1 string to match.
    if (memcmp(Name.data()+0, "_Null_unspecified", 17) != 0)
      break;
    return AttributeCommonInfo::AT_TypeNullUnspecified;	 // "_Null_unspecified"
  case 19:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__unsafe_unretained", 19) != 0)
      break;
    return AttributeCommonInfo::AT_ObjCInertUnsafeUnretained;	 // "__unsafe_unretained"
  case 20:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__single_inheritance", 20) != 0)
      break;
    return AttributeCommonInfo::AT_MSInheritance;	 // "__single_inheritance"
  case 21:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__virtual_inheritance", 21) != 0)
      break;
    return AttributeCommonInfo::AT_MSInheritance;	 // "__virtual_inheritance"
  case 22:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__multiple_inheritance", 22) != 0)
      break;
    return AttributeCommonInfo::AT_MSInheritance;	 // "__multiple_inheritance"
  case 23:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__arm_locally_streaming", 23) != 0)
      break;
    return AttributeCommonInfo::AT_ArmLocallyStreaming;	 // "__arm_locally_streaming"
  case 25:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__unspecified_inheritance", 25) != 0)
      break;
    return AttributeCommonInfo::AT_MSInheritance;	 // "__unspecified_inheritance"
  case 26:	 // 1 string to match.
    if (memcmp(Name.data()+0, "__arm_streaming_compatible", 26) != 0)
      break;
    return AttributeCommonInfo::AT_ArmStreamingCompatible;	 // "__arm_streaming_compatible"
  }
  } else if (AttributeCommonInfo::AS_Pragma == Syntax) {
  switch (Name.size()) {
  default: break;
  case 4:	 // 1 string to match.
    if (memcmp(Name.data()+0, "loop", 4) != 0)
      break;
    return AttributeCommonInfo::AT_LoopHint;	 // "loop"
  case 6:	 // 1 string to match.
    if (memcmp(Name.data()+0, "unroll", 6) != 0)
      break;
    return AttributeCommonInfo::AT_LoopHint;	 // "unroll"
  case 8:	 // 1 string to match.
    if (memcmp(Name.data()+0, "nounroll", 8) != 0)
      break;
    return AttributeCommonInfo::AT_LoopHint;	 // "nounroll"
  case 14:	 // 1 string to match.
    if (memcmp(Name.data()+0, "unroll_and_jam", 14) != 0)
      break;
    return AttributeCommonInfo::AT_LoopHint;	 // "unroll_and_jam"
  case 16:	 // 1 string to match.
    if (memcmp(Name.data()+0, "nounroll_and_jam", 16) != 0)
      break;
    return AttributeCommonInfo::AT_LoopHint;	 // "nounroll_and_jam"
  }
  } else if (AttributeCommonInfo::AS_HLSLAnnotation == Syntax) {
  switch (Name.size()) {
  default: break;
  case 8:	 // 1 string to match.
    if (memcmp(Name.data()+0, "register", 8) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLResourceBinding;	 // "register"
  case 10:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ackoffset", 9) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLPackOffset;	 // "packoffset"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "v_groupid", 9) != 0)
        break;
      return AttributeCommonInfo::AT_HLSLSV_GroupID;	 // "sv_groupid"
    }
    break;
  case 13:	 // 1 string to match.
    if (memcmp(Name.data()+0, "sv_groupindex", 13) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLSV_GroupIndex;	 // "sv_groupindex"
  case 16:	 // 1 string to match.
    if (memcmp(Name.data()+0, "sv_groupthreadid", 16) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLSV_GroupThreadID;	 // "sv_groupthreadid"
  case 19:	 // 1 string to match.
    if (memcmp(Name.data()+0, "sv_dispatchthreadid", 19) != 0)
      break;
    return AttributeCommonInfo::AT_HLSLSV_DispatchThreadID;	 // "sv_dispatchthreadid"
  }
  }
  return AttributeCommonInfo::UnknownAttribute;
}
