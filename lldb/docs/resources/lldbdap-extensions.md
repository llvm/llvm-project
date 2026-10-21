# LLDB-DAP Extensions

lldb-dap implements the
[Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/specification)
(DAP). This page specifies what lldb-dap adds to the protocol, such as new requests and additional fields in protocol types.

Conventions:

- New requests should be prefixed with `__lldb_`.
- Added properties on standard messages are prefixed with `$__lldb_` if the field
  would not make sense for any other debugger (e.g. `$__lldb_version`);
  otherwise, they use plain names (e.g. `compileUnitId`).

## Custom Requests

### CompileUnits Request

The request is sent from the client to the debug adapter to list the compile
units of a module. Use it with `compileUnitId` of a [`StackFrame`](#stackframe)
and the `id` of a [`Module`](#module).

```typescript
interface CompileUnitsRequest extends Request {
  command: 'compileUnits';

  arguments: CompileUnitsArguments;
}

interface CompileUnitsArguments {
  /**
   * The UUID of the module.
   */
  moduleId: string;

  /**
   * The compile units to return. If empty or omitted, all compile units of
   * the module are returned.
   * @format uint32
   */
  compileUnitIds?: number[];
}

interface CompileUnitsResponse extends Response {
  body: {
    compileUnits: CompileUnit[];
  };
}

interface CompileUnit {
  /**
   * The ID of the compile unit, unique within its module.
   * @format uint32
   */
  id: number;

  /**
   * The path of the compile unit.
   */
  compileUnitPath: string;

  /**
   * True if the compile unit was compiled with optimization, false if it
   * was unoptimized or unknown. Optional so a debugger can omit it when
   * unknown, in which case clients should treat it as unknown.
   */
  isOptimized?: boolean;
}
```

### ModuleSymbols Request

The request is sent from the client to the debug adapter to list the symbols of
a module. It is only available if the capability `supportsModuleSymbolsRequest`
is true.

```typescript
interface ModuleSymbolsRequest extends Request {
  command: '__lldb_moduleSymbols';

  arguments: ModuleSymbolsArguments;
}

interface ModuleSymbolsArguments {
  /**
   * The UUID of the module.
   */
  moduleId: string;

  /**
   * The name of the module.
   */
  moduleName: string;

  /**
   * The index of the first symbol to return. If omitted, starts at 0.
   * @format uint32
   */
  startIndex?: number;

  /**
   * The number of symbols to return. If count is missing or 0, all symbols
   * are returned.
   * @format uint32
   */
  count?: number;
}

interface ModuleSymbolsResponse extends Response {
  body: {
    /**
    * All symbols or range of symbols.
    */
    symbols: Symbol[];
  };
}

interface Symbol {
  /**
   * The ID of the symbol.
   * @format uint32
   */
  id: number;

  /**
   * The name of the symbol.
   */
  name: string;

  /**
   * The size of the symbol in bytes.
   * @format uint64
   */
  size: number;

  /**
   * The symbol has debug information.
   */
  isDebug: boolean;

  /**
   * The symbol was synthesized by the debugger.
   */
  isSynthetic: boolean;

  /**
   * The symbol is visible outside of its module.
   */
  isExternal: boolean;

  /**
   * The `lldb::SymbolType`, e.g. 'Code', 'Trampoline', 'Data'.
   */
  type: string;

  /**
   * The address of the symbol in the module file.
   * @format uint64
   */
  fileAddress: number;

  /**
   * The address of the symbol in the process. Omitted if the module is not
   * loaded.
   * @format uint64
   */
  loadAddress?: number;
}
```

## Requests with Additional Properties

### Initialize Request

```typescript
interface InitializeRequestArguments {
  /**
   * Whether the adapter sources `lldbinit` files when initializing the
   * debugger.
   */
  $__lldb_sourceInitFile?: boolean;
}
```

```typescript
interface InitializeResponse extends Response {
  body?: Capabilities;
}

interface Capabilities {
  /**
   * The version of lldb same as (`version` command).
   */
  $__lldb_version?: string;

  /**
   * The debug adapter supports the `__lldb_moduleSymbols` request.
   */
  supportsModuleSymbolsRequest?: boolean;
}
```

### ExceptionInfo Request

```typescript
interface ExceptionInfoResponse extends Response {
  body: {
    /**
     * The raw JSON from `SBThread::GetStopReasonExtendedInfoAsJSON`.
     */
    $__lldb_stopReasonExtendedInfo?: any;

    /**
     * The raw JSON from `SBProcess::GetExtendedCrashInformation`.
     */
    $__lldb_extendedCrashInformation?: any;
  };
}
```

## Types with Additional Properties

### Module

```typescript
interface Module {
  /**
   * The size of the module's debug information as a human readable string,
   * e.g. '12.3 MB'.
   */
  debugInfoSize?: string;
}
```

### StackFrame

```typescript
interface StackFrame {
  /**
   * The compile unit of the frame. Use with `moduleId` in the `compileUnits`
   * request.
   * @format uint32
   */
  compileUnitId?: number;
}
```
