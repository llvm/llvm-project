
%extend lldb::SBSaveCoreOptions {
#ifdef SWIGPYTHON
    %pythoncode%{
        '''Add a thread to the SaveCoreOptions thread list, and follow its children N pointers deep, adding each memory region to the SaveCoreOptions Memory region list.'''
        def save_thread_with_heaps(self, thread, num_pointers_deep = 3):
            self.AddThread(thread)
            frame = thread.GetFrameAtIndex(0)
            process = thread.GetProcess()
            queue = []
            for var in frame.locals:
                type = var.GetType()
                if type.IsPointerType() or type.IsReferenceType():
                    queue.append(var.Dereference())
            
            while (num_pointers_deep > 0 and len(queue) > 0):
                queue_size = len(queue)
                for i in range(0, queue_size):
                    var = queue.pop(0)
                    var_type = var.GetType()
                    addr = var.GetAddress().GetOffset()
                    memory_region = lldb.SBMemoryRegionInfo(None, addr, addr + var_type.GetByteSize(), False)
                    self.AddMemoryRegionToSave(memory_region)
                    # We only check for direct pointer children T*, should probably scan 
                    # internal to the children themselves. 
                    for x in var.children:
                        if x.TypeIsPointerType():
                            queue.append(x.Dereference())

                num_pointers_deep -= 1
    %}
#endif
}
