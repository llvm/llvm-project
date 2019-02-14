import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestTargetSourceMap(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_source_map(self):
        """Test target.source-map' functionality."""
        # Set the target soure map to map "./" to the current test directory
        src_dir = self.getSourceDir()
        src_path = os.path.join(src_dir, "main.c")
        yaml_path = os.path.join(src_dir, "a.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)

        # Set a breakpoint before we remap source and verify that it fails
        bp = target.BreakpointCreateByLocation(src_path, 2)
        self.assertTrue(bp.GetNumLocations() == 0,
                        "make sure no breakpoints were resolved without map")
        src_map_cmd = 'settings set target.source-map . "%s"' % (src_dir)
        self.dbg.HandleCommand(src_map_cmd)

        # Set a breakpoint after we remap source and verify that it succeeds
        bp = target.BreakpointCreateByLocation(src_path, 2)
        self.assertTrue(bp.GetNumLocations() == 1,
                        "make sure breakpoint was resolved with map")

        # Now make sure that we can actually FIND the source file using this
        # remapping:
        retval = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("source list -f main.c -l 2", retval)
        self.assertTrue(retval.Succeeded(), "source list didn't succeed.")
        self.assertTrue(retval.GetOutput() != None, "We got no ouput from source list")
        self.assertTrue("return" in retval.GetOutput(), "We didn't find the source file...")
        
