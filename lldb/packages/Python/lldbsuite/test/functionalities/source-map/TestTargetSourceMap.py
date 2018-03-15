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
        obj_path = self.getBuildArtifact(yaml_base)
        self.yaml2obj(yaml_path, obj_path)

        def cleanup():
            if os.path.exists(obj_path):
                os.unlink(obj_path)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)

        # Set a breakpoint before we remap source and verify that it fails
        bp = target.BreakpointCreateByLocation(src_path, 2)
        self.assertTrue(bp.GetNumLocations() == 0,
                        "make sure no breakpoints were resolved without map")
        src_map_cmd = 'settings set target.source-map ./ "%s"' % (src_dir)
        self.dbg.HandleCommand(src_map_cmd)

        # Set a breakpoint after we remap source and verify that it succeeds
        bp = target.BreakpointCreateByLocation(src_path, 2)
        self.assertTrue(bp.GetNumLocations() == 1,
                        "make sure breakpoint was resolved with map")
