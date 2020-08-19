from .builder import Builder


class BuilderDarwin(Builder):
    def buildDsym(self,
                  sender=None,
                  architecture=None,
                  compiler=None,
                  dictionary=None,
                  testdir=None,
                  testname=None):
        """Build the binaries with dsym debug info."""
        commands = []
        commands.append(
            self.getMake(testdir, testname) + [
                "MAKE_DSYM=YES",
                self.getArchSpec(architecture),
                self.getCCSpec(compiler),
                self.getDsymutilSpec(),
                self.getSDKRootSpec(),
                self.getModuleCacheSpec(), "all",
                self.getCmdLine(dictionary)
            ])

        self.runBuildCommands(commands, sender=sender)

        # True signifies that we can handle building dsym.
        return True
