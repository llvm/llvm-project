LLDB RPC Upstreaming Design Doc
===============================

This document aims to explain the general structure of the upstreaming patches for adding LLDB RPC. The 2 primary concepts explained here will be:

* How LLDB RPC is used
* How the ``lldb-rpc-gen`` works and what it outputs

LLDB RPC
*********

LLDB RPC is a framework by which processes can communicate with LLDB out of process while maintaining compatibility with the SB API. More details are explained in the `RFC<https://discourse.llvm.org/t/rfc-upstreaming-lldb-rpc/85804>`_ for upstreaming LLDB RPC, but the main focus in this doc for this section will be how exactly the code is structured for the PRs that will upstream this code.

The ``lldb-rpc-gen`` tool
*************************

``lldb-rpc-gen`` is the tool that generates the main client and server interfaces for LLDB RPC. It is a ``ClangTool`` that reads all SB API header files and their functions and outputs the client/server interfaces and certain other pieces of code, such as RPC-specfic versions of Python bindings used for the test suite. There's 3 main components behind ``lldb-rpc-gen``:

1. The ``lldb-rpc-gen`` tool itself, which contains the main driver that uses the ``ClangTool``.
2. The code that generates all interfaces, which we call "emitters". All generated code for the interfaces are in C++, so the server side has one emitter for its generated source code and another for its generated header code. The client side has the same.
3. All common code shared between all emitters, such as helper methods and information about exceptions to take when emitting.

There are currently 2 PRs up for upstreaming RPC:
- One that adds the ``lldb-rpc-gen`` tool and its common code: https://github.com/llvm/llvm-project/pull/138031
- One that adds the RPC client-side interface code emitters: https://github.com/llvm/llvm-project/pull/147655

The `current PR<https://github.com/llvm/llvm-project/pull/136748>`_ up for upstreaming LLDB RPC upstreams a subset of the code used for the tool. It upstreams the ``lldb-rpc-gen`` tool and all code needed for the server side emitters. Here's an example of what ``lldb-rpc-gen`` will output for the server side interface:

Input
-----

We'll use ``SBDebugger::CreateTarget(const char *filename)`` as an example. ``lldb-rpc-gen`` will read this method from ``SBDebugger.h``. The output is as follows.

Server-side Output
******************

Source Code Output
------------------

Server-side Source Code
~~~~~~~~~~~~~~~~~~~~~~~
::

   bool rpc_server::_ZN4lldb10SBDebugger12CreateTargetEPKc::HandleRPCCall(rpc_common::Connection &connection, RPCStream &send, RPCStream &response) {
   // 1) Make local storage for incoming function arguments
   lldb::SBDebugger *this_ptr = nullptr;
   rpc_common::ConstCharPointer filename;
   // 2) Decode all function arguments
   this_ptr = RPCServerObjectDecoder<lldb::SBDebugger>(send, rpc_common::RPCPacket::ValueType::Argument);
   if (!this_ptr)
   return false;
   if (!RPCValueDecoder(send, rpc_common::RPCPacket::ValueType::Argument, filename))
   return false;
   // 3) Call the method and encode the return value
   lldb::SBTarget && __result = this_ptr->CreateTarget(filename.c_str());
   RPCServerObjectEncoder(response, rpc_common::RPCPacket::ValueType::ReturnValue, std::move(__result));
   return true;
   }

Function signature
~~~~~~~~~~~~~~~~~~

All server-side source code functions have a function signature that take the format ``bool rpc_server::<mangled-function-name>::HandleRPCCall(rpc_common::Connection &connection, RPCStream &send, RPCStream &response)``. The mangled name is used in order to differentiate between overloaded methods. Here the ``connection`` is what's maintained between the client and server. The ``send`` variable is a byte stream that carries information sent from the client. ``response`` is also a byte stream that will be populated with the return value obtained from the call into the SB API function that will be sent back to the client.

For the client-side sources, the function signature is identical to that of what the signature looks like in the main SB API.

Local variable storage
~~~~~~~~~~~~~~~~~~~~~~

First, variables are created to hold all arguments coming in from the client side. These variables will be a pointer for the SB API class in question, and corresponding variables for all parameters that the function has. Since this signature for ``SBDebugger::CreateTarget()`` only has one parameter, a ``const char *``, 2 local variables get created. A pointer for an ``SBDebugger`` object, and an ``RPCCommon::ConstCharPointer`` for the ``const char * filename`` parameter. The ``ConstCharPointer`` is a class backed by ``std::string`` in the main RPC core code.

Incoming stream decoding
~~~~~~~~~~~~~~~~~~~~~~~~

Following this, ``RPCServerObjectDecoder`` is used to decode the ``send`` byte stream. In this case, we're decoding this stream into the ``SBDebugger`` pointer we created earlier. We then decode the ``send`` stream again to obtain the ``const char * filename`` sent by the client. Each decoded argument from the client is checked for validity and the function will exit early if any are invalid.

SB API function call
~~~~~~~~~~~~~~~~~~~~

Once all arguments have been decoded, the underlying SB API function called with the decoded arguments. ``RPCServerObjectEncoder`` is then used to encode the return value from the SB API call into the ``response`` stream, and this is then sent back to the client.

Header Code Output
------------------
::

   class _ZN4lldb10SBDebugger12CreateTargetEPKc : public rpc_common::RPCFunctionInstance {
   public:
   _ZN4lldb10SBDebugger12CreateTargetEPKc() : RPCFunctionInstance("_ZN4lldb10SBDebugger12CreateTargetEPKc") {}
   ~_ZN4lldb10SBDebugger12CreateTargetEPKc() override {}
   bool HandleRPCCall(rpc_common::Connection &connection, rpc_common::RPCStream &send, rpc_common::RPCStream &response) override;
   };

Class definition and ``HandleRPCCall``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All RPC server side functions are subclasses of ``RPCFunctionInstance``. Each class will then define their ``HandleRPCCall`` function that is seen in the source code above. This subclassing and ``HandleRPCCall`` definition is what is emitted in the header code for server.

Client-side Output
******************

Client-side Source Code
~~~~~~~~~~~~~~~~~~~~~~~
::

   lldb_rpc::SBTarget lldb_rpc::SBDebugger::CreateTarget(const char * filename) {
   // 1) Perform setup
   // Storage for return value
   lldb_rpc::SBTarget __result = {};
   // Deriving connection from this.
   rpc_common::ConnectionSP connection_sp = ObjectRefGetConnectionSP();
   if (!connection_sp) return __result;
   // RPC Communication setup
   static RPCFunctionInfo g_func("_ZN4lldb10SBDebugger12CreateTargetEPKc");
   RPCStream send;
   RPCStream response;
   g_func.Encode(send);
   RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, *this);
   RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, filename);
   // 2) Send RPC call
   if (!connection_sp->SendRPCCallAndWaitForResponse(send, response))
   return __result;
   // 3) Decode return values
   RPCValueDecoder(response, rpc_common::RPCPacket::ValueType::ReturnValue, __result);
   return __result;
   }

Function signature
~~~~~~~~~~~~~~~~~~

For the client-side sources, the function signature is almost always identical to that of what the
signature looks like in the main SB API, with the namespace changing from ``lldb`` to ``lldb_rpc``. For some methods, the function signature might need to change to prepend an RPC connection as the first argument. This happens in the event that the function is static. Since RPC functions usually derive their connection from their instance, static functions must be given a connection as they have no instance to derive one from.

Return Value Storage
~~~~~~~~~~~~~~~~~~~~

We first need to create storage for the return value. For this method we return an ``SBTarget``, so we need to create an ``lldb_rpc::SBTarget`` for the return value and initialize it to an empty value. Since we are on the client-side, all instances where we use SB API classes will be from the ``lldb_rpc`` namespace.

Obtaining RPC Connection
~~~~~~~~~~~~~~~~~~~~~~~~

We then need to obtain the RPC connection. In this case, we obtain the connection by deriving from the ``SBDebugger`` instance that would've been created prior to this function call by using ``ObjectRefGetConnectionSP()``. If this connection is invalid then we return an empty value.

Encoding RPC Stream Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the mangled function name for the function information, we encode the streams used to send the function's parameters and receive the return info from the server-side call. For this function, we need to encode the pointer to the ``SBDebugger()`` instance itself, and the ``const char *filename*`` parameter as well.

Sending Encoded Information to Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the information has been encoded, ``SendRPCCallAndWaitForResponse(send, response)`` is used to send the information to the server-side, where the underlying call to ``lldb::SBDebugger::CreateTarget(const char **)`` will be made. If this call failed, then an empty value will be returned.

Decoding Information Received from Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the call to send the encoded information was successful, then we need to decode what the server sent in response. This decoded value is then placed in the return value storage that we set up earlier and is then returned to the connected client itself.

Header Code Output
------------------

::
   lldb_rpc::SBTarget CreateTarget(const char * filename);

The output for the header code on the client-side is significantly simpler. Similar to the function signature for the client-side sources, the header file output is just the function signature for each method. This means that it will match the original SB API function signature, with the only exceptions being for static methods as stated above for the source code function signature.

``lldb-rpc-gen`` emitters
*****************************

The bulk of the code is generated using the emitters. For the server side, we have ``RPCServerSourceEmitter`` and ``RPCServerHeaderEmitter``. The former handles generation of the source code and the latter handles generation of the header code seen above. Similarly for the client side, we have ``RPCLibrarySourceEmitter`` and ``RPCLibraryHeaderEmitter``. Similar the server emitters, these emitter handle generating the source and header code for the client side, respectively.

Emitters largely have similar structure. Constituent sections of code, such as function headers, function bodies and others are typically given their own method. As an example, the function to emit a function header is ``EmitFunctionHeader()`` and the function to emit a function body is ``EmitFunctionBody()``. Information that will be written to the output file is written using ``EmitLine()``, which uses ``llvm::raw_string_ostreams`` and is defined in ``RPCCommon.h``.

Since this is a ``ClangTool``, information about each method is obtained from Clang itself and stored in the ``Method`` struct located in ``RPCCommon.h`` in ``lldb-rpc-gen``'s directory. ``Method`` is used for simplicity and abstraction, and other information that would be needed from the SB API is obtained from Clang directly.

Testing
*******

The RPC interface and the code emitters are tested in 2 main ways:

- The RPC client and server interfaces are tested by running the full LLDB SB API test suite against liblldbrpc. Using this, Python acts as the client binary connecting to RPC, and all SB API calls from API tests will go through the RPC client/server flow as described above. This has its own ninja target, ``check-lldb-rpc``.
- The RPC client and server emitters are tested using shell tests where FileCheck checks the output of the emitters against a set of heuristics that we have. Currently, these shell tests exist for the client side emitters as they have more heuristics than the server-side emitters.
