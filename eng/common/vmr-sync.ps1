<#
.SYNOPSIS

This script is used for synchronizing the current repository into a local VMR.
It pulls the current repository's code into the specified VMR directory for local testing or
Source-Build validation.

.DESCRIPTION

The tooling used for synchronization will clone the VMR repository into a temporary folder if
it does not already exist. These clones can be reused in future synchronizations, so it is
recommended to dedicate a folder for this to speed up re-runs.

.EXAMPLE
  Synchronize current repository into a local VMR:
    ./vmr-sync.ps1 -vmrDir "$HOME/repos/dotnet" -tmpDir "$HOME/repos/tmp"

.PARAMETER tmpDir
Required. Path to the temporary folder where repositories will be cloned

.PARAMETER vmrBranch
Optional. Branch of the 'dotnet/dotnet' repo to synchronize. The VMR will be checked out to this branch

.PARAMETER azdevPat
Optional. Azure DevOps PAT to use for cloning private repositories.

.PARAMETER vmrDir
Optional. Path to the dotnet/dotnet repository. When null, gets cloned to the temporary folder

.PARAMETER debugOutput
Optional. Enables debug logging in the darc vmr command.

.PARAMETER ci
Optional. Denotes that the script is running in a CI environment.
#>
param (
  [Parameter(Mandatory=$true, HelpMessage="Path to the temporary folder where repositories will be cloned")]
  [string][Alias('t', 'tmp')]$tmpDir,
  [string][Alias('b', 'branch')]$vmrBranch,
  [string]$remote,
  [string]$azdevPat,
  [string][Alias('v', 'vmr')]$vmrDir,
  [switch]$ci,
  [switch]$debugOutput
)

function Fail {
  Write-Host "> $($args[0])" -ForegroundColor 'Red'
}

function Highlight {
  Write-Host "> $($args[0])" -ForegroundColor 'Cyan'
}

$verbosity = 'verbose'
if ($debugOutput) {
  $verbosity = 'debug'
}
# Validation

if (-not $tmpDir) {
  Fail "Missing -tmpDir argument. Please specify the path to the temporary folder where the repositories will be cloned"
  exit 1
}

# Sanitize the input

if (-not $vmrDir) {
  $vmrDir = Join-Path $tmpDir 'dotnet'
}

if (-not (Test-Path -Path $tmpDir -PathType Container)) {
  New-Item -ItemType Directory -Path $tmpDir | Out-Null
}

# Prepare the VMR

if (-not (Test-Path -Path $vmrDir -PathType Container)) {
  Highlight "Cloning 'dotnet/dotnet' into $vmrDir.."
  git clone https://github.com/dotnet/dotnet $vmrDir

  if ($vmrBranch) {
    git -C $vmrDir switch -c $vmrBranch
  }
}
else {
  if ((git -C $vmrDir diff --quiet) -eq $false) {
    Fail "There are changes in the working tree of $vmrDir. Please commit or stash your changes"
    exit 1
  }

  if ($vmrBranch) {
    Highlight "Preparing $vmrDir"
    git -C $vmrDir checkout $vmrBranch
    git -C $vmrDir pull
  }
}

Set-StrictMode -Version Latest

# Prepare darc

Highlight 'Installing .NET, preparing the tooling..'
. .\eng\common\tools.ps1
$dotnetRoot = InitializeDotNetCli -install:$true
$darc = Get-Darc
$dotnet = "$dotnetRoot\dotnet.exe"

Highlight "Starting the synchronization of VMR.."

# Synchronize the VMR
$darcArgs = (
  "vmr", "forwardflow",
  "--tmp", $tmpDir,
  "--$verbosity",
  $vmrDir
)

if ($ci) {
  $darcArgs += ("--ci")
}

if ($azdevPat) {
  $darcArgs += ("--azdev-pat", $azdevPat)
}

& "$darc" $darcArgs

if ($LASTEXITCODE -eq 0) {
  Highlight "Synchronization succeeded"
}
else {
  Fail "Synchronization of repo to VMR failed!"
  Fail "'$vmrDir' is left in its last state (re-run of this script will reset it)."
  Fail "Please inspect the logs which contain path to the failing patch file (use -debugOutput to get all the details)."
  Fail "Once you make changes to the conflicting VMR patch, commit it locally and re-run this script."
  exit 1
}
