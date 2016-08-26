# Change Notes

## Summary

This document describes the DarwinLog logging feature.

## StructuredDataDarwinLog feature

The DarwinLog feature supports logging os_log*() and NSLog() messages
to the command-line lldb console, as well as making those messages
available to LLDB clients via the event system.  Starting with fall
2016 OSes, Apple platforms introduce a new fire-hose, stream-style
logging system where the bulk of the log processing happens on the log
consumer side.  This reduces logging impact on the system when there
are no consumers, making it cheaper to include logging at all times.
However, it also increases the work needed on the consumer end when
log messages are desired.

The debugserver binary has been modified to support collection of
os_log*()/NSLog() messages, selection of which messages appear in the
stream, and fine-grained filtering of what gets passed on to the LLDB
client.  DarwinLog also tracks the activity chain (i.e. os_activity()
hierarchy) in effect at the time the log messages were issued.  The
user is able to configure a number of aspects related to the
formatting of the log message header fields.

The DarwinLog support is written in a way which should support the
lldb client side on non-Apple clients talking to an Apple device or
macOS system; hence, the plugin support is built into all LLDB
clients, not just those built on an Apple platform.

StructuredDataDarwinLog implements the 'DarwinLog' feature type, and
the plugin name for it shows up as 'darwin-log'.

The user interface to the darwin-log support is via the following:

* 'plugin structured-data darwin-log enable' command

  This is the main entry point for enabling the command.  It can be
  set before launching a process or while the process is running.
  If the user wants to squelch seeing info-level or debug-level
  messages, which is the default behavior, then the enable command
  must be made prior to launching the process; otherwise, the
  info-level and debug-level messages will always show up.  Also,
  there is a similar "echo os_log()/NSLog() messages to target
  process stderr" mechanism which is properly disabled when enabling
  the DarwinLog support prior to launch.  This cannot be squelched
  if enabling DarwinLog after launch.

  See the help for this command.  There are a number of options
  to shrink or expand the number of messages that are processed
  on the remote side and sent over to the client, and other
  options to control the formatting of messages displayed.

  This command is sticky.  Once enabled, it will stay enabled for
  future process launches.

* 'plugin structured-data darwin-log disable' command

  Executing this command disables os_log() capture in the currently
  running process and signals LLDB to stop attempting to launch
  new processes with DarwinLog support enabled.

* 'settings set \
  plugin.structured-data.darwin-log.enable-on-startup'

  and

  'settings set \
  plugin.structured-data.darwin-log.auto-enable-options -- {options}'

  When enable-on-startup is set to true, then LLDB will automatically
  enable DarwinLog on startup of relevant processes.  It will use the
  content provided in the auto-enable-options settings as the
  options to pass to the enable command.

  Note the '--' required after auto-enable-command.  That is necessary
  for raw commands like settings set.  The '--' will not become part
  of the options for the enable command.

### Message flow and related performance considerations

os_log()-style collection is not free.  The more data that must be
processed, the slower it will be.  There are several knobs available
to the developer to limit how much data goes through the pipe, and how
much data ultimately goes over the wire to the LLDB client.  The
user's goal should be to ensure he or she only collects as many log
messages are needed, but no more.

The flow of data looks like the following:

1. Data comes into debugserver from the low-level OS facility that
   receives log messages.  The data that comes through this pipe can
   be limited or expanded by the '--debug', '--info' and
   '--all-processes' options of the 'plugin structured-data darwin-log
   enable' command.  options.  Exclude as many categories as possible
   here (also the default).  The knobs here are very coarse - for
   example, whether to include os_log_info()-level or
   os_log_debug()-level info, or to include callstacks in the log
   message event data.

2. The debugserver process filters the messages that arrive through a
   message log filter that may be fully customized by the user.  It
   works similar to a rules-based packet filter: a set of rules are
   matched against the log message, each rule tried in sequential
   order.  The first rule that matches then either accepts or rejects
   the message.  If the log message does not match any rule, then the
   message gets the no-match (i.e. fall-through) action.  The no-match
   action defaults to accepting but may be set to reject.

   Filters can be added via the enable command's '--filter
   {filter-spec}' option.  Filters are added in order, and multiple
   --filter entries can be provided to the enable command.

   Filters take the following form:

   {action} {attribute} {op}

   {action} :=
       accept |
       reject

   {attribute} :=
       category       |   // The log message category
       subsystem      |   // The log message subsystem}
       activity       |   // The child-most activity in force
                          // at the time the message was logged.
       activity-chain |   // The complete activity chain, specified
                          // as {parent-activity}:{child-activity}:
                          // {grandchild-activity}
       message        |   // The fully expanded message contents.
                          // Note this one is expensive because it
                          // requires expanding the message.  Avoid
                          // this if possible, or add it further
                          // down the filter chain.

   {op} :=
              match {exact-match-text} |
              regex {search-regex}        // uses C++ std::regex
                                          // ECMAScript variant.

e.g.
   --filter "accept subsystem match com.example.mycompany.myproduct"
   --filter "accept subsystem regex com.example.+"
   --filter "reject category regex spammy-system-[[:digit:]]+"

3. Messages that are accepted by the log message filter get sent to
   the lldb client, where they are mapped to the
   StructuredDataDarwinLog plugin.  By default, command-line lldb will
   issue a Process-level event containing the log message content, and
   will request the plugin to print the message if the plugin is
   enabled to do so.

### Log message display

Several settings control aspects of displaying log messages in
command-line LLDB.  See the enable command's help for a description
of these.


