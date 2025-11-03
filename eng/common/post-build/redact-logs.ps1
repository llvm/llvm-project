[CmdletBinding(PositionalBinding=$False)]
param(
  [Parameter(Mandatory=$true, Position=0)][string] $InputPath,
  [Parameter(Mandatory=$true)][string] $BinlogToolVersion,
  [Parameter(Mandatory=$false)][string] $DotnetPath,
  [Parameter(Mandatory=$false)][string] $PackageFeed = 'https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-public/nuget/v3/index.json',
  # File with strings to redact - separated by newlines.
  #  For comments start the line with '# ' - such lines are ignored 
  [Parameter(Mandatory=$false)][string] $TokensFilePath,
  [Parameter(ValueFromRemainingArguments=$true)][String[]]$TokensToRedact,
  [Parameter(Mandatory=$false)][string] $runtimeSourceFeed,
  [Parameter(Mandatory=$false)][string] $runtimeSourceFeedKey
)

try {
  $ErrorActionPreference = 'Stop'
  Set-StrictMode -Version 2.0

  # `tools.ps1` checks $ci to perform some actions. Since the post-build
  # scripts don't necessarily execute in the same agent that run the
  # build.ps1/sh script this variable isn't automatically set.
  $ci = $true
  $disableConfigureToolsetImport = $true
  . $PSScriptRoot\..\tools.ps1

  $packageName = 'binlogtool'

  $dotnet = $DotnetPath

  if (!$dotnet) {
    $dotnetRoot = InitializeDotNetCli -install:$true
    $dotnet = "$dotnetRoot\dotnet.exe"
  }
  
  $toolList = & "$dotnet" tool list -g

  if ($toolList -like "*$packageName*") {
    & "$dotnet" tool uninstall $packageName -g
  }

  $toolPath  = "$PSScriptRoot\..\..\..\.tools"
  $verbosity = 'minimal'
  
  New-Item -ItemType Directory -Force -Path $toolPath
  
  Push-Location -Path $toolPath

  try {
    Write-Host "Installing Binlog redactor CLI..."
    Write-Host "'$dotnet' new tool-manifest"
    & "$dotnet" new tool-manifest
    Write-Host "'$dotnet' tool install $packageName --local --add-source '$PackageFeed' -v $verbosity --version $BinlogToolVersion"
    & "$dotnet" tool install $packageName --local --add-source "$PackageFeed" -v $verbosity --version $BinlogToolVersion

    if (Test-Path $TokensFilePath) {
        Write-Host "Adding additional sensitive data for redaction from file: " $TokensFilePath
        $TokensToRedact += Get-Content -Path $TokensFilePath | Foreach {$_.Trim()} | Where { $_ -notmatch "^# " }
    }

    $optionalParams = [System.Collections.ArrayList]::new()
  
    Foreach ($p in $TokensToRedact)
    {
      if($p -match '^\$\(.*\)$')
      {
        Write-Host ("Ignoring token {0} as it is probably unexpanded AzDO variable"  -f $p)
      }          
      elseif($p)
      {
        $optionalParams.Add("-p:" + $p) | Out-Null
      }
    }

    & $dotnet binlogtool redact --input:$InputPath --recurse --in-place `
      @optionalParams

    if ($LastExitCode -ne 0) {
      Write-PipelineTelemetryError -Category 'Redactor' -Type 'warning' -Message "Problems using Redactor tool (exit code: $LastExitCode). But ignoring them now."
    }
  }
  finally {
    Pop-Location
  }

  Write-Host 'done.'
} 
catch {
  Write-Host $_
  Write-PipelineTelemetryError -Category 'Redactor' -Message "There was an error while trying to redact logs. Error: $_"
  ExitWithExitCode 1
}
