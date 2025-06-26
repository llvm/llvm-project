# This script is used to install the .NET SDK.
# It will also invoke the SDK with any provided arguments.

. $PSScriptRoot\tools.ps1
$dotnetRoot = InitializeDotNetCli -install:$true

# Invoke acquired SDK with args if they are provided
if ($args.count -gt 0) {
  $env:DOTNET_NOLOGO=1
  & "$dotnetRoot\dotnet.exe" $args
}
