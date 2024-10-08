#extend lldb::SBSaveCoreOptions {
#ifdef SWIGPYTHON
    %pythoncode% {
        '''Add a thread to the SaveCoreOptions thread list, and follow it's children N pointers deep, adding each memory region to the SaveCoreOptions Memory region list.'''
        def save_thread_with_heaps(self, thread, num_heaps_deep = 3):
            self.AddThread(thread)
            frame = thread.GetFrameAtIndex(0)
            process = thread.GetProcess()
            queue = []
            for var in frame.locals:
                if var.TypeIsPointerType():
                    queue.append(var.Dereference())
            
            while (num_heaps_deep > 0 and len(queue) > 0):
                queue_size = len(queue)
                for i in range(0, queue_size):
                    var = queue.pop(0)
                    memory_region = lldb.SBMemoryRegionInfo()
                    process.GetMemoryRegionInfo(var.GetAddress().GetOffset(), memory_region)
                    self.AddMemoryRegionToSave(memory_region)
                    /* 
                    We only check for direct pointer children T*, should probably scan 
                    internal to the children themselves. 
                    */
                    for x in var.children:
                        if x.TypeIsPointerType():
                            queue.append(x.Dereference())

                num_heaps_deep -= 1
}
#endif SWIGPYTHON
