class BuildError(Exception):

    def __init__(self, called_process_error):
        super(BuildError, self).__init__("Error when building test subject")
        self.command = called_process_error.lldb_extensions.get(
            "command", "<command unavailable>")
        self.build_error = called_process_error.lldb_extensions.get(
            "stderr_content", "<error output unavailable>")

    def __str__(self):
        return self.format_build_error(self.command, self.build_error)

    @staticmethod
    def format_build_error(command, command_output):
        return "Error when building test subject.\n\nBuild Command:\n{}\n\nBuild Command Output:\n{}".format(
            command, command_output)
