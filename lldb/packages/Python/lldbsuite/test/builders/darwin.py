from .builder import Builder

from lldbsuite.test import configuration


class BuilderDarwin(Builder):
    def getExtraMakeArgs(self):
        """
        Helper function to return extra argumentsfor the make system. This
        method is meant to be overridden by platform specific builders.
        """
        args = dict()

        if configuration.dsymutil:
            args['DSYMUTIL'] = configuration.dsymutil

        # Return extra args as a formatted string.
        return ' '.join(
            {'{}="{}"'.format(key, value)
             for key, value in args.items()})

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
                self.getExtraMakeArgs(),
                self.getSDKRootSpec(),
                self.getModuleCacheSpec(), "all",
                self.getCmdLine(dictionary)
            ])

        self.runBuildCommands(commands, sender=sender)

        # True signifies that we can handle building dsym.
        return True
