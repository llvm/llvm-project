# JavaScript Reference

LLDB has extensive support for interacting with JavaScript through the V8
JavaScript engine. This document describes how to use JavaScript scripting
within LLDB and provides reference documentation for the JavaScript API.

## Using JavaScript in LLDB

LLDB's JavaScript support is built on top of the V8 JavaScript engine, the same
engine that powers Node.js and Chrome. This provides full ES2020+ language
support with modern JavaScript features.

### Interactive JavaScript

JavaScript can be run interactively in LLDB. First, set JavaScript as the script language, then use the `script` command:

```
(lldb) settings set script-lang javascript
(lldb) script
>>> let message = "Hello from JavaScript!";
>>> console.log(message);
Hello from JavaScript!
>>> lldb.debugger.GetVersionString()
lldb version 18.0.0
```

### Running JavaScript from Files

You can execute JavaScript files using the `command script import` command:

```
(lldb) command script import /path/to/myscript.js
```

The JavaScript file will be executed in the current LLDB context with access
to all LLDB APIs.

### Example JavaScript Script

Here's a simple example that demonstrates using the LLDB JavaScript API:

```javascript
// Get the current debugger instance
let debugger = lldb.debugger;

// Get the current target
let target = debugger.GetSelectedTarget();

// Get the current process
let process = target.GetProcess();

// Get the selected thread
let thread = process.GetSelectedThread();

// Get the selected frame
let frame = thread.GetSelectedFrame();

// Evaluate an expression
let result = frame.EvaluateExpression("myVariable");
console.log("Value:", result.GetValue());

// Print all local variables
let variables = frame.GetVariables(true, true, false, false);
for (let i = 0; i < variables.GetSize(); i++) {
  let variable = variables.GetValueAtIndex(i);
  console.log(variable.GetName() + " = " + variable.GetValue());
}
```

## The JavaScript API

The JavaScript API provides access to all of LLDB's Script Bridge (SB) API
classes. These classes are automatically available in the `lldb` module when
running JavaScript within LLDB.

### Global Objects

* `lldb`: The main LLDB module containing all SB API classes
* `lldb.debugger`: The current debugger instance (shortcut to avoid passing
  debugger around)
* `console`: Standard JavaScript console object for logging

### Available Classes

The JavaScript API includes all of LLDB's SB API classes, like `SBDebugger`,
`SBTarget`, etc.

For complete documentation of all classes and their methods, refer to the
[C++ API documentation](https://lldb.llvm.org/cpp_reference/namespacelldb.html),
as the JavaScript API mirrors the C++ API closely.

### Console Output

JavaScript scripts can use the standard `console` object for output:

```javascript
console.log("Informational message");
console.error("Error message");
console.warn("Warning message");
```

Output from `console.log()` and other console methods will be displayed in
the LLDB command output.

## Building LLDB with JavaScript Support

### Prerequisites

To build LLDB with JavaScript support, you need:

* [V8 JavaScript Engine](https://v8.dev) (version 8.0 or later recommended)
* [SWIG](http://swig.org/) 4 or later (for generating language bindings)
* All standard LLDB build dependencies (see [build documentation](../resources/build.rst))

### Installing V8

The V8 JavaScript engine must be installed on your system. Installation methods
vary by platform:

**Ubuntu/Debian:**

```bash
$ sudo apt-get install libv8-dev
```

After installation, V8 will typically be installed in:
- Headers: `/usr/include/v8/` or `/usr/include/`
- Libraries: `/usr/lib/x86_64-linux-gnu/libv8.so` (or similar for your architecture)

You can verify the installation with:
```bash
$ dpkg -L libv8-dev | grep -E '(include|lib)'
```

**macOS (using Homebrew):**

```bash
$ brew install v8
```

After installation, you can find the paths with:
```bash
$ brew info v8
```

Homebrew typically installs to `/opt/homebrew/` (Apple Silicon) or `/usr/local/` (Intel).

**Building V8 from source:**

If V8 is not available as a package for your platform, you can build it from
source. Follow the instructions at https://v8.dev/docs/build

### CMake Configuration

To enable JavaScript support when building LLDB, add the following CMake
options:

```bash
$ cmake -G Ninja \
    -DLLDB_ENABLE_JAVASCRIPT=ON \
    [other cmake options] \
    /path/to/llvm-project/llvm
```

The `LLDB_ENABLE_JAVASCRIPT` flag enables JavaScript scripting support. If
V8 is installed via a package manager in standard system locations, CMake
should auto-detect it. If CMake cannot find V8, you can specify the paths
manually:

```bash
$ cmake -G Ninja \
    -DLLDB_ENABLE_JAVASCRIPT=ON \
    -DV8_INCLUDE_DIR=/path/to/v8/include \
    -DV8_LIBRARIES=/path/to/v8/lib/libv8.so \
    [other cmake options] \
    /path/to/llvm-project/llvm
```

where:
* `V8_INCLUDE_DIR`: Path to V8 header files
* `V8_LIBRARIES`: Path to V8 library files

### Verifying JavaScript Support

After building LLDB with JavaScript support, you can verify it's working:

```
$ lldb
(lldb) settings set script-lang javascript
(lldb) script
>>> console.log("JavaScript is working!")
JavaScript is working!
>>> lldb.debugger.GetVersionString()
lldb version 18.0.0
```

If JavaScript support is not enabled, you'll see an error message when trying
to set the script language to JavaScript.

### Build Example

Here's a complete example of building LLDB with JavaScript support from scratch:

```bash
# Clone the LLVM project
$ git clone https://github.com/llvm/llvm-project.git

# Create build directory
$ mkdir llvm-build && cd llvm-build

# Configure with JavaScript support
$ cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;lldb" \
    -DLLDB_ENABLE_JAVASCRIPT=ON \
    -DV8_INCLUDE_DIR=/usr/include/v8 \
    -DV8_LIBRARIES=/usr/lib/x86_64-linux-gnu/libv8.so \
    ../llvm-project/llvm

# Build LLDB
$ ninja lldb

# Test JavaScript support
$ ./bin/lldb -o "settings set script-lang javascript" -o "script -e \"console.log('Hello!')\"" -o "quit"
```

## Differences from Python API and JavaScript Environment

Important differences to understand:

**Not a Node.js Environment:**

LLDB's JavaScript environment uses the V8 engine but is **not** Node.js. This means:

* **No module system**: `import`, `require()`, and `module.exports` are not available
* **No event loop**: Asynchronous operations like `setTimeout`, `setInterval`, `Promise.then()` callbacks are not supported
* **Limited global APIs**: Only specific functions are implemented:
  * `console.log()`, `console.error()`, `console.warn()` for output
  * `lldb` global object for LLDB API access
  * Standard JavaScript language features (ES2020+)

**Module Access:**

In Python, you typically import with `import lldb`. In JavaScript, `lldb`
is automatically available as a global object without any import statement.

Scripts should be written as self-contained synchronous code that directly uses the
`lldb` global object.

## Known Limitations

The JavaScript support in LLDB is not as extensive as Python. The
following features are not yet implemented:

* Custom breakpoint callbacks in JavaScript
* Custom watchpoint callbacks in JavaScript
* Some advanced type mapping and conversions

## Additional Resources

* [LLDB C++ API Reference](https://lldb.llvm.org/cpp_reference/namespacelldb.html)
* [V8 JavaScript Engine Documentation](https://v8.dev/docs)
* [LLDB Python Reference](python-reference.html) (similar concepts apply to JavaScript)
