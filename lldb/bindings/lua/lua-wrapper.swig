%header %{

template <typename T>
void
PushSBClass(lua_State* L, T* obj);

%}

%wrapper %{

// This function is called from Lua::CallBreakpointCallback
SWIGEXPORT llvm::Expected<bool>
LLDBSwigLuaBreakpointCallbackFunction
(
   lua_State *L,
   lldb::StackFrameSP stop_frame_sp,
   lldb::BreakpointLocationSP bp_loc_sp
)
{
   lldb::SBFrame sb_frame(stop_frame_sp);
   lldb::SBBreakpointLocation sb_bp_loc(bp_loc_sp);

   // Push the Lua wrappers
   PushSBClass(L, &sb_frame);
   PushSBClass(L, &sb_bp_loc);

   // Call into the Lua callback passing 'sb_frame' and 'sb_bp_loc'.
   // Expects a boolean return.
   if (lua_pcall(L, 2, 1, 0) != LUA_OK) {
      llvm::Error E = llvm::make_error<llvm::StringError>(
            llvm::formatv("{0}\n", lua_tostring(L, -1)),
            llvm::inconvertibleErrorCode());
      // Pop error message from the stack.
      lua_pop(L, 1);
      return std::move(E);
   }

   // Boolean return from the callback
   bool stop = lua_toboolean(L, -1);
   lua_pop(L, 1);

   return stop;
}


%}
