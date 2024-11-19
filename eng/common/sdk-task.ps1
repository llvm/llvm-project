[CmdletBinding(PositionalBinding=$false)]
Param(
  [string] $configuration = 'Debug',
  [string] $task,
  [string] $verbosity = 'minimal',
  [string] $msbuildEngine = $null,
  [switch] $restore,
  [switch] $prepareMachine,
  [switch] $help,
  [Parameter(ValueFromRemainingArguments=$true)][String[]]$properties
)

$ci = $true
$binaryLog = $true
$warnAsError = $true

. $PSScriptRoot\tools.ps1

function Print-Usage() {
  Write-Host "Common settings:"
  Write-Host "  -task <value>           Name of Arcade task (name of a project in SdkTasks directory of the Arcade SDK package)"
  Write-Host "  -restore                Restore dependencies"
  Write-Host "  -verbosity <value>      Msbuild verbosity: q[uiet], m[inimal], n[ormal], d[etailed], and diag[nostic]"
  Write-Host "  -help                   Print help and exit"
  Write-Host ""

  Write-Host "Advanced settings:"
  Write-Host "  -prepareMachine         Prepare machine for CI run"
  Write-Host "  -msbuildEngine <value>  Msbuild engine to use to run build ('dotnet', 'vs', or unspecified)."
  Write-Host ""
  Write-Host "Command line arguments not listed above are passed thru to msbuild."
}

function Build([string]$target) {
  $logSuffix = if ($target -eq 'Execute') { '' } else { ".$target" }
  $log = Join-Path $LogDir "$task$logSuffix.binlog"
  $outputPath = Join-Path $ToolsetDir "$task\"

  MSBuild $taskProject `
    /bl:$log `
    /t:$target `
    /p:Configuration=$configuration `
    /p:RepoRoot=$RepoRoot `
    /p:BaseIntermediateOutputPath=$outputPath `
    /v:$verbosity `
    @properties
}

try {
  if ($help -or (($null -ne $properties) -and ($properties.Contains('/help') -or $properties.Contains('/?')))) {
    Print-Usage
    exit 0
  }

  if ($task -eq "") {
    Write-PipelineTelemetryError -Category 'Build' -Message "Missing required parameter '-task <value>'"
    Print-Usage
    ExitWithExitCode 1
  }

  if( $msbuildEngine -eq "vs") {
    # Ensure desktop MSBuild is available for sdk tasks.
    if( -not ($GlobalJson.tools.PSObject.Properties.Name -contains "vs" )) {
      $GlobalJson.tools | Add-Member -Name "vs" -Value (ConvertFrom-Json "{ `"version`": `"16.5`" }") -MemberType NoteProperty
    }
    if( -not ($GlobalJson.tools.PSObject.Properties.Name -match "xcopy-msbuild" )) {
      $GlobalJson.tools | Add-Member -Name "xcopy-msbuild" -Value "17.12.0" -MemberType NoteProperty
    }
    if ($GlobalJson.tools."xcopy-msbuild".Trim() -ine "none") {
        $xcopyMSBuildToolsFolder = InitializeXCopyMSBuild $GlobalJson.tools."xcopy-msbuild" -install $true
    }
    if ($xcopyMSBuildToolsFolder -eq $null) {
      throw 'Unable to get xcopy downloadable version of msbuild'
    }

    $global:_MSBuildExe = "$($xcopyMSBuildToolsFolder)\MSBuild\Current\Bin\MSBuild.exe"
  }

  $taskProject = GetSdkTaskProject $task
  if (!(Test-Path $taskProject)) {
    Write-PipelineTelemetryError -Category 'Build' -Message "Unknown task: $task"
    ExitWithExitCode 1
  }

  if ($restore) {
    Build 'Restore'
  }

  Build 'Execute'
}
catch {
  Write-Host $_.ScriptStackTrace
  Write-PipelineTelemetryError -Category 'Build' -Message $_
  ExitWithExitCode 1
}

ExitWithExitCode 0
