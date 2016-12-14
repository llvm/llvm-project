# Change Notes

## Overview

This document describes an infrastructural feature called Structured
Data plugins.  See the `DarwinLog.md` doc for a description of one
such plugin that makes use of this feature.

## StructuredDataPlugin

StructuredDataPlugin instances have the following characteristics:

* Each plugin instance is bound to a single Process instance.

* Each StructuredData feature has a type name that identifies the
  feature. For instance, the type name for the DarwinLog feature is
  "DarwinLog". This feature type name is used in various places.

* The process monitor reports the list of supported StructuredData
  features advertised by the process monitor. Process goes through the
  list of supported feature type names, and asks each known
  StructuredDataPlugin if it can handle the feature. The first plugin
  that supports the feature is mapped to that Process instance for
  that feature.  Plugins are only mapped when the process monitor
  advertises that a feature is supported.

* The feature may send asynchronous messages in StructuredData format
  to the Process instance. Process instances route the asynchronous
  structured data messages to the plugin mapped to that feature type,
  if one exists.

* Plugins can request that the Process instance forward on
  configuration data to the process monitor if the plugin needs/wants
  to configure the feature. Plugins may call the new Process method

  ```C++
  virtual Error
  ConfigureStructuredData(const ConstString &type_name,
                          const StructuredData::ObjectSP &config_sp)
  ```

  where `type_name` is the feature name and `config_sp` points to the
  configuration structured data, which may be nullptr.

* Plugins for features present in a process are notified when modules
  are loaded into the Process instance via this StructuredDataPlugin
  method:

  ```C++
  virtual void
  ModulesDidLoad(Process &process, ModuleList &module_list);
  ```

* Plugins may optionally broadcast their received structured data as
  an LLDB process-level event via the following new Process call:

  ```C++
  void
  BroadcastStructuredData(const StructuredData::ObjectSP &object_sp,
                          const lldb::StructuredDataPluginSP &plugin_sp);
  ```

  IDE clients might use this feature to receive information about the
  process as it is running to monitor memory usage, CPU usage, and
  logging.

  Internally, the event type created is an instance of
  EventDataStructuredData.

* In the case where a plugin chooses to broadcast a received
  StructuredData event, the command-line LLDB Debugger instance
  listens for them. The Debugger instance then gives the plugin an
  opportunity to display info to either the debugger output or error
  stream at a time that is safe to write to them. The plugin can
  choose to display something appropriate regarding the structured
  data that time.

* Plugins can provide a ProcessLaunchInfo filter method when the
  plugin is registered.  If such a filter method is provided, then
  when a process is about to be launched for debugging, the filter
  callback is invoked, given both the launch info and the target.  The
  plugin may then alter the launch info if needed to better support
  the feature of the plugin.

* The plugin is entirely independent of the type of Process-derived
  class that it is working with. The only requirements from the
  process monitor are the following feature-agnostic elements:

  * Provide a way to discover features supported by the process
    monitor for the current process.

  * Specify the list of supported feature type names to Process.
    The process monitor does this by calling the following new
    method on Process:

    ```C++
    void
    MapSupportedStructuredDataPlugins(const StructuredData::Array
                                      &supported_type_names)
    ```

    The `supported_type_names` specifies an array of string entries,
    where each entry specifies the name of a StructuredData feature.

  * Provide a way to forward on configuration data for a feature type
    to the process monitor.  This is the manner by which LLDB can
    configure a feature, perhaps based on settings or commands from
    the user.  The following virtual method on Process (described
    earlier) does the job:

    ```C++
    virtual Error
    ConfigureStructuredData(const ConstString &type_name,
                            const StructuredData::ObjectSP &config_sp)
    ```

  * Listen for asynchronous structured data packets from the process
    monitor, and forward them on to Process via this new Process
    member method:

    ```C++
    bool
    RouteAsyncStructuredData(const StructuredData::ObjectSP object_sp)
    ```

* StructuredData producers must send their top-level data as a
  Dictionary type, with a key called 'type' specifying a string value,
  where the value is equal to the StructuredData feature/type name
  previously advertised. Everything else about the content of the
  dictionary is entirely up to the feature.

* StructuredDataPlugin commands show up under `plugin structured-data
  plugin-name`.

* StructuredDataPlugin settings show up under
  `plugin.structured-data.{plugin-name}`.
