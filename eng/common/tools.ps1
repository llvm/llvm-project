# Initialize variables if they aren't already defined.
# These may be defined as parameters of the importing script, or set after importing this script.

# CI mode - set to true on CI server for PR validation build or official build.
[bool]$ci = if (Test-Path variable:ci) { $ci } else { $false }

# Build configuration. Common values include 'Debug' and 'Release', but the repository may use other names.
[string]$configuration = if (Test-Path variable:configuration) { $configuration } else { 'Debug' }

# Set to true to opt out of outputting binary log while running in CI
[bool]$excludeCIBinarylog = if (Test-Path variable:excludeCIBinarylog) { $excludeCIBinarylog } else { $false }

# Set to true to output binary log from msbuild. Note that emitting binary log slows down the build.
[bool]$binaryLog = if (Test-Path variable:binaryLog) { $binaryLog } else { $ci -and !$excludeCIBinarylog }

# Set to true to use the pipelines logger which will enable Azure logging output.
# https://github.com/Microsoft/azure-pipelines-tasks/blob/master/docs/authoring/commands.md
# This flag is meant as a temporary opt-opt for the feature while validate it across
# our consumers. It will be deleted in the future.
[bool]$pipelinesLog = if (Test-Path variable:pipelinesLog) { $pipelinesLog } else { $ci }

# Turns on machine preparation/clean up code that changes the machine state (e.g. kills build processes).
[bool]$prepareMachine = if (Test-Path variable:prepareMachine) { $prepareMachine } else { $false }

# True to restore toolsets and dependencies.
[bool]$restore = if (Test-Path variable:restore) { $restore } else { $true }

# Adjusts msbuild verbosity level.
[string]$verbosity = if (Test-Path variable:verbosity) { $verbosity } else { 'minimal' }

# Set to true to reuse msbuild nodes. Recommended to not reuse on CI.
[bool]$nodeReuse = if (Test-Path variable:nodeReuse) { $nodeReuse } else { !$ci }

# Configures warning treatment in msbuild.
[bool]$warnAsError = if (Test-Path variable:warnAsError) { $warnAsError } else { $true }

# Specifies which msbuild engine to use for build: 'vs', 'dotnet' or unspecified (determined based on presence of tools.vs in global.json).
[string]$msbuildEngine = if (Test-Path variable:msbuildEngine) { $msbuildEngine } else { $null }

# True to attempt using .NET Core already that meets requirements specified in global.json
# installed on the machine instead of downloading one.
[bool]$useInstalledDotNetCli = if (Test-Path variable:useInstalledDotNetCli) { $useInstalledDotNetCli } else { $true }

# Enable repos to use a particular version of the on-line dotnet-install scripts.
#    default URL: https://builds.dotnet.microsoft.com/dotnet/scripts/v1/dotnet-install.ps1
[string]$dotnetInstallScriptVersion = if (Test-Path variable:dotnetInstallScriptVersion) { $dotnetInstallScriptVersion } else { 'v1' }

# True to use global NuGet cache instead of restoring packages to repository-local directory.
[bool]$useGlobalNuGetCache = if (Test-Path variable:useGlobalNuGetCache) { $useGlobalNuGetCache } else { !$ci }

# True to exclude prerelease versions Visual Studio during build
[bool]$excludePrereleaseVS = if (Test-Path variable:excludePrereleaseVS) { $excludePrereleaseVS } else { $false }

# An array of names of processes to stop on script exit if prepareMachine is true.
$processesToStopOnExit = if (Test-Path variable:processesToStopOnExit) { $processesToStopOnExit } else { @('msbuild', 'dotnet', 'vbcscompiler') }

$disableConfigureToolsetImport = if (Test-Path variable:disableConfigureToolsetImport) { $disableConfigureToolsetImport } else { $null }

set-strictmode -version 2.0
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# If specifies, provides an alternate path for getting .NET Core SDKs and Runtimes. This script will still try public sources first.
[string]$runtimeSourceFeed = if (Test-Path variable:runtimeSourceFeed) { $runtimeSourceFeed } else { $null }
# Base-64 encoded SAS token that has permission to storage container described by $runtimeSourceFeed
[string]$runtimeSourceFeedKey = if (Test-Path variable:runtimeSourceFeedKey) { $runtimeSourceFeedKey } else { $null }

# True when the build is running within the VMR.
[bool]$fromVMR = if (Test-Path variable:fromVMR) { $fromVMR } else { $false }

function Create-Directory ([string[]] $path) {
    New-Item -Path $path -Force -ItemType 'Directory' | Out-Null
}

function Unzip([string]$zipfile, [string]$outpath) {
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

# This will exec a process using the console and return it's exit code.
# This will not throw when the process fails.
# Returns process exit code.
function Exec-Process([string]$command, [string]$commandArgs) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $command
  $startInfo.Arguments = $commandArgs
  $startInfo.UseShellExecute = $false
  $startInfo.WorkingDirectory = Get-Location

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  $process.Start() | Out-Null

  $finished = $false
  try {
    while (-not $process.WaitForExit(100)) {
      # Non-blocking loop done to allow ctr-c interrupts
    }

    $finished = $true
    return $global:LASTEXITCODE = $process.ExitCode
  }
  finally {
    # If we didn't finish then an error occurred or the user hit ctrl-c.  Either
    # way kill the process
    if (-not $finished) {
      $process.Kill()
    }
  }
}

# Take the given block, print it, print what the block probably references from the current set of
# variables using low-effort string matching, then run the block.
#
# This is intended to replace the pattern of manually copy-pasting a command, wrapping it in quotes,
# and printing it using "Write-Host". The copy-paste method is more readable in build logs, but less
# maintainable and less reliable. It is easy to make a mistake and modify the command without
# properly updating the "Write-Host" line, resulting in misleading build logs. The probability of
# this mistake makes the pattern hard to trust when it shows up in build logs. Finding the bug in
# existing source code can also be difficult, because the strings are not aligned to each other and
# the line may be 300+ columns long.
#
# By removing the need to maintain two copies of the command, Exec-BlockVerbosely avoids the issues.
#
# In Bash (or any posix-like shell), "set -x" prints usable verbose output automatically.
# "Set-PSDebug" appears to be similar at first glance, but unfortunately, it isn't very useful: it
# doesn't print any info about the variables being used by the command, which is normally the
# interesting part to diagnose.
function Exec-BlockVerbosely([scriptblock] $block) {
  Write-Host "--- Running script block:"
  $blockString = $block.ToString().Trim()
  Write-Host $blockString

  Write-Host "--- List of variables that might be used:"
  # For each variable x in the environment, check the block for a reference to x via simple "$x" or
  # "@x" syntax. This doesn't detect other ways to reference variables ("${x}" nor "$variable:x",
  # among others). It only catches what this function was originally written for: simple
  # command-line commands.
  $variableTable = Get-Variable |
    Where-Object {
      $blockString.Contains("`$$($_.Name)") -or $blockString.Contains("@$($_.Name)")
    } |
    Format-Table -AutoSize -HideTableHeaders -Wrap |
    Out-String
  Write-Host $variableTable.Trim()

  Write-Host "--- Executing:"
  & $block
  Write-Host "--- Done running script block!"
}

# createSdkLocationFile parameter enables a file being generated under the toolset directory
# which writes the sdk's location into. This is only necessary for cmd --> powershell invocations
# as dot sourcing isn't possible.
function InitializeDotNetCli([bool]$install, [bool]$createSdkLocationFile) {
  if (Test-Path variable:global:_DotNetInstallDir) {
    return $global:_DotNetInstallDir
  }

  # Disable first run since we do not need all ASP.NET packages restored.
  $env:DOTNET_NOLOGO=1

  # Disable telemetry on CI.
  if ($ci) {
    $env:DOTNET_CLI_TELEMETRY_OPTOUT=1
  }

  # Find the first path on %PATH% that contains the dotnet.exe
  if ($useInstalledDotNetCli -and (-not $globalJsonHasRuntimes) -and ($env:DOTNET_INSTALL_DIR -eq $null)) {
    $dotnetExecutable = GetExecutableFileName 'dotnet'
    $dotnetCmd = Get-Command $dotnetExecutable -ErrorAction SilentlyContinue

    if ($dotnetCmd -ne $null) {
      $env:DOTNET_INSTALL_DIR = Split-Path $dotnetCmd.Path -Parent
    }
  }

  $dotnetSdkVersion = $GlobalJson.tools.dotnet

  # Use dotnet installation specified in DOTNET_INSTALL_DIR if it contains the required SDK version,
  # otherwise install the dotnet CLI and SDK to repo local .dotnet directory to avoid potential permission issues.
  if ((-not $globalJsonHasRuntimes) -and (-not [string]::IsNullOrEmpty($env:DOTNET_INSTALL_DIR)) -and (Test-Path(Join-Path $env:DOTNET_INSTALL_DIR "sdk\$dotnetSdkVersion"))) {
    $dotnetRoot = $env:DOTNET_INSTALL_DIR
  } else {
    $dotnetRoot = Join-Path $RepoRoot '.dotnet'

    if (-not (Test-Path(Join-Path $dotnetRoot "sdk\$dotnetSdkVersion"))) {
      if ($install) {
        InstallDotNetSdk $dotnetRoot $dotnetSdkVersion
      } else {
        Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "Unable to find dotnet with SDK version '$dotnetSdkVersion'"
        ExitWithExitCode 1
      }
    }

    $env:DOTNET_INSTALL_DIR = $dotnetRoot
  }

  # Creates a temporary file under the toolset dir.
  # The following code block is protecting against concurrent access so that this function can
  # be called in parallel.
  if ($createSdkLocationFile) {
    do {
      $sdkCacheFileTemp = Join-Path $ToolsetDir $([System.IO.Path]::GetRandomFileName())
    }
    until (!(Test-Path $sdkCacheFileTemp))
    Set-Content -Path $sdkCacheFileTemp -Value $dotnetRoot

    try {
      Move-Item -Force $sdkCacheFileTemp (Join-Path $ToolsetDir 'sdk.txt')
    } catch {
      # Somebody beat us
      Remove-Item -Path $sdkCacheFileTemp
    }
  }

  # Add dotnet to PATH. This prevents any bare invocation of dotnet in custom
  # build steps from using anything other than what we've downloaded.
  # It also ensures that VS msbuild will use the downloaded sdk targets.
  $env:PATH = "$dotnetRoot;$env:PATH"

  # Make Sure that our bootstrapped dotnet cli is available in future steps of the Azure Pipelines build
  Write-PipelinePrependPath -Path $dotnetRoot

  Write-PipelineSetVariable -Name 'DOTNET_NOLOGO' -Value '1'

  return $global:_DotNetInstallDir = $dotnetRoot
}

function Retry($downloadBlock, $maxRetries = 5) {
  $retries = 1

  while($true) {
    try {
      & $downloadBlock
      break
    }
    catch {
      Write-PipelineTelemetryError -Category 'InitializeToolset' -Message $_
    }

    if (++$retries -le $maxRetries) {
      $delayInSeconds = [math]::Pow(2, $retries) - 1 # Exponential backoff
      Write-Host "Retrying. Waiting for $delayInSeconds seconds before next attempt ($retries of $maxRetries)."
      Start-Sleep -Seconds $delayInSeconds
    }
    else {
      Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "Unable to download file in $maxRetries attempts."
      break
    }
  }
}

function GetDotNetInstallScript([string] $dotnetRoot) {
  $installScript = Join-Path $dotnetRoot 'dotnet-install.ps1'
  $shouldDownload = $false
  
  if (!(Test-Path $installScript)) {
    $shouldDownload = $true
  } else {
    # Check if the script is older than 30 days
    $fileAge = (Get-Date) - (Get-Item $installScript).LastWriteTime
    if ($fileAge.Days -gt 30) {
      Write-Host "Existing install script is too old, re-downloading..."
      $shouldDownload = $true
    }
  }
  
  if ($shouldDownload) {
    Create-Directory $dotnetRoot
    $ProgressPreference = 'SilentlyContinue' # Don't display the console progress UI - it's a huge perf hit
    $uri = "https://builds.dotnet.microsoft.com/dotnet/scripts/$dotnetInstallScriptVersion/dotnet-install.ps1"

    Retry({
      Write-Host "GET $uri"
      Invoke-WebRequest $uri -UseBasicParsing -OutFile $installScript
    })
  }

  return $installScript
}

function InstallDotNetSdk([string] $dotnetRoot, [string] $version, [string] $architecture = '', [switch] $noPath) {
  InstallDotNet $dotnetRoot $version $architecture '' $false $runtimeSourceFeed $runtimeSourceFeedKey -noPath:$noPath
}

function InstallDotNet([string] $dotnetRoot,
  [string] $version,
  [string] $architecture = '',
  [string] $runtime = '',
  [bool] $skipNonVersionedFiles = $false,
  [string] $runtimeSourceFeed = '',
  [string] $runtimeSourceFeedKey = '',
  [switch] $noPath) {

  $dotnetVersionLabel = "'sdk v$version'"

  if ($runtime -ne '' -and $runtime -ne 'sdk') {
    $runtimePath = $dotnetRoot
    $runtimePath = $runtimePath + "\shared"
    if ($runtime -eq "dotnet") { $runtimePath = $runtimePath + "\Microsoft.NETCore.App" }
    if ($runtime -eq "aspnetcore") { $runtimePath = $runtimePath + "\Microsoft.AspNetCore.App" }
    if ($runtime -eq "windowsdesktop") { $runtimePath = $runtimePath + "\Microsoft.WindowsDesktop.App" }
    $runtimePath = $runtimePath + "\" + $version
  
    $dotnetVersionLabel = "runtime toolset '$runtime/$architecture v$version'"

    if (Test-Path $runtimePath) {
      Write-Host "  Runtime toolset '$runtime/$architecture v$version' already installed."
      $installSuccess = $true
      Exit
    }
  }

  $installScript = GetDotNetInstallScript $dotnetRoot
  $installParameters = @{
    Version = $version
    InstallDir = $dotnetRoot
  }

  if ($architecture) { $installParameters.Architecture = $architecture }
  if ($runtime) { $installParameters.Runtime = $runtime }
  if ($skipNonVersionedFiles) { $installParameters.SkipNonVersionedFiles = $skipNonVersionedFiles }
  if ($noPath) { $installParameters.NoPath = $True }

  $variations = @()
  $variations += @($installParameters)

  $dotnetBuilds = $installParameters.Clone()
  $dotnetbuilds.AzureFeed = "https://ci.dot.net/public"
  $variations += @($dotnetBuilds)

  if ($runtimeSourceFeed) {
    $runtimeSource = $installParameters.Clone()
    $runtimeSource.AzureFeed = $runtimeSourceFeed
    if ($runtimeSourceFeedKey) {
      $decodedBytes = [System.Convert]::FromBase64String($runtimeSourceFeedKey)
      $decodedString = [System.Text.Encoding]::UTF8.GetString($decodedBytes)
      $runtimeSource.FeedCredential = $decodedString
    }
    $variations += @($runtimeSource)
  }

  $installSuccess = $false
  foreach ($variation in $variations) {
    if ($variation | Get-Member AzureFeed) {
      $location = $variation.AzureFeed
    } else {
      $location = "public location";
    }
    Write-Host "  Attempting to install $dotnetVersionLabel from $location."
    try {
      & $installScript @variation
      $installSuccess = $true
      break
    }
    catch {
      Write-Host "  Failed to install $dotnetVersionLabel from $location."
    }
  }
  if (-not $installSuccess) {
    Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "Failed to install $dotnetVersionLabel from any of the specified locations."
    ExitWithExitCode 1
  }
}

#
# Locates Visual Studio MSBuild installation.
# The preference order for MSBuild to use is as follows:
#
#   1. MSBuild from an active VS command prompt
#   2. MSBuild from a compatible VS installation
#   3. MSBuild from the xcopy tool package
#
# Returns full path to msbuild.exe.
# Throws on failure.
#
function InitializeVisualStudioMSBuild([bool]$install, [object]$vsRequirements = $null) {
  if (-not (IsWindowsPlatform)) {
    throw "Cannot initialize Visual Studio on non-Windows"
  }

  if (Test-Path variable:global:_MSBuildExe) {
    return $global:_MSBuildExe
  }

  # Minimum VS version to require.
  $vsMinVersionReqdStr = '17.7'
  $vsMinVersionReqd = [Version]::new($vsMinVersionReqdStr)

  # If the version of msbuild is going to be xcopied,
  # use this version. Version matches a package here:
  # https://dev.azure.com/dnceng/public/_artifacts/feed/dotnet-eng/NuGet/Microsoft.DotNet.Arcade.MSBuild.Xcopy/versions/18.0.0
  $defaultXCopyMSBuildVersion = '18.0.0'

  if (!$vsRequirements) {
    if (Get-Member -InputObject $GlobalJson.tools -Name 'vs') {
      $vsRequirements = $GlobalJson.tools.vs
    }
    else {
      $vsRequirements = New-Object PSObject -Property @{ version = $vsMinVersionReqdStr }
    }
  }
  $vsMinVersionStr = if ($vsRequirements.version) { $vsRequirements.version } else { $vsMinVersionReqdStr }
  $vsMinVersion = [Version]::new($vsMinVersionStr)

  # Try msbuild command available in the environment.
  if ($env:VSINSTALLDIR -ne $null) {
    $msbuildCmd = Get-Command 'msbuild.exe' -ErrorAction SilentlyContinue
    if ($msbuildCmd -ne $null) {
      # Workaround for https://github.com/dotnet/roslyn/issues/35793
      # Due to this issue $msbuildCmd.Version returns 0.0.0.0 for msbuild.exe 16.2+
      $msbuildVersion = [Version]::new((Get-Item $msbuildCmd.Path).VersionInfo.ProductVersion.Split([char[]]@('-', '+'))[0])

      if ($msbuildVersion -ge $vsMinVersion) {
        return $global:_MSBuildExe = $msbuildCmd.Path
      }

      # Report error - the developer environment is initialized with incompatible VS version.
      throw "Developer Command Prompt for VS $($env:VisualStudioVersion) is not recent enough. Please upgrade to $vsMinVersionStr or build from a plain CMD window"
    }
  }

  # Locate Visual Studio installation or download x-copy msbuild.
  $vsInfo = LocateVisualStudio $vsRequirements
  if ($vsInfo -ne $null -and $env:ForceUseXCopyMSBuild -eq $null) {
    # Ensure vsInstallDir has a trailing slash
    $vsInstallDir = Join-Path $vsInfo.installationPath "\"
    $vsMajorVersion = $vsInfo.installationVersion.Split('.')[0]

    InitializeVisualStudioEnvironmentVariables $vsInstallDir $vsMajorVersion
  } else {
    if (Get-Member -InputObject $GlobalJson.tools -Name 'xcopy-msbuild') {
      $xcopyMSBuildVersion = $GlobalJson.tools.'xcopy-msbuild'
      $vsMajorVersion = $xcopyMSBuildVersion.Split('.')[0]
    } else {
      #if vs version provided in global.json is incompatible (too low) then use the default version for xcopy msbuild download
      if($vsMinVersion -lt $vsMinVersionReqd){
        Write-Host "Using xcopy-msbuild version of $defaultXCopyMSBuildVersion since VS version $vsMinVersionStr provided in global.json is not compatible"
        $xcopyMSBuildVersion = $defaultXCopyMSBuildVersion
        $vsMajorVersion = $xcopyMSBuildVersion.Split('.')[0]
      }
      else{
        # If the VS version IS compatible, look for an xcopy msbuild package
        # with a version matching VS.
        # Note: If this version does not exist, then an explicit version of xcopy msbuild
        # can be specified in global.json. This will be required for pre-release versions of msbuild.
        $vsMajorVersion = $vsMinVersion.Major
        $vsMinorVersion = $vsMinVersion.Minor
        $xcopyMSBuildVersion = "$vsMajorVersion.$vsMinorVersion.0"
      }
    }

    $vsInstallDir = $null
    if ($xcopyMSBuildVersion.Trim() -ine "none") {
        $vsInstallDir = InitializeXCopyMSBuild $xcopyMSBuildVersion $install
        if ($vsInstallDir -eq $null) {
            throw "Could not xcopy msbuild. Please check that package 'Microsoft.DotNet.Arcade.MSBuild.Xcopy @ $xcopyMSBuildVersion' exists on feed 'dotnet-eng'."
        }
    }
    if ($vsInstallDir -eq $null) {
      throw 'Unable to find Visual Studio that has required version and components installed'
    }
  }

  $msbuildVersionDir = if ([int]$vsMajorVersion -lt 16) { "$vsMajorVersion.0" } else { "Current" }

  $local:BinFolder = Join-Path $vsInstallDir "MSBuild\$msbuildVersionDir\Bin"
  $local:Prefer64bit = if (Get-Member -InputObject $vsRequirements -Name 'Prefer64bit') { $vsRequirements.Prefer64bit } else { $false }
  if ($local:Prefer64bit -and (Test-Path(Join-Path $local:BinFolder "amd64"))) {
    $global:_MSBuildExe = Join-Path $local:BinFolder "amd64\msbuild.exe"
  } else {
    $global:_MSBuildExe = Join-Path $local:BinFolder "msbuild.exe"
  }

  return $global:_MSBuildExe
}

function InitializeVisualStudioEnvironmentVariables([string] $vsInstallDir, [string] $vsMajorVersion) {
  $env:VSINSTALLDIR = $vsInstallDir
  Set-Item "env:VS$($vsMajorVersion)0COMNTOOLS" (Join-Path $vsInstallDir "Common7\Tools\")

  $vsSdkInstallDir = Join-Path $vsInstallDir "VSSDK\"
  if (Test-Path $vsSdkInstallDir) {
    Set-Item "env:VSSDK$($vsMajorVersion)0Install" $vsSdkInstallDir
    $env:VSSDKInstall = $vsSdkInstallDir
  }
}

function InstallXCopyMSBuild([string]$packageVersion) {
  return InitializeXCopyMSBuild $packageVersion -install $true
}

function InitializeXCopyMSBuild([string]$packageVersion, [bool]$install) {
  $packageName = 'Microsoft.DotNet.Arcade.MSBuild.Xcopy'
  $packageDir = Join-Path $ToolsDir "msbuild\$packageVersion"
  $packagePath = Join-Path $packageDir "$packageName.$packageVersion.nupkg"

  if (!(Test-Path $packageDir)) {
    if (!$install) {
      return $null
    }

    Create-Directory $packageDir

    Write-Host "Downloading $packageName $packageVersion"
    $ProgressPreference = 'SilentlyContinue' # Don't display the console progress UI - it's a huge perf hit
    Retry({
      Invoke-WebRequest "https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-eng/nuget/v3/flat2/$packageName/$packageVersion/$packageName.$packageVersion.nupkg" -UseBasicParsing -OutFile $packagePath
    })

    if (!(Test-Path $packagePath)) {
      Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "See https://dev.azure.com/dnceng/internal/_wiki/wikis/DNCEng%20Services%20Wiki/1074/Updating-Microsoft.DotNet.Arcade.MSBuild.Xcopy-WAS-RoslynTools.MSBuild-(xcopy-msbuild)-generation?anchor=troubleshooting for help troubleshooting issues with XCopy MSBuild"
      throw
    }
    Unzip $packagePath $packageDir
  }

  return Join-Path $packageDir 'tools'
}

#
# Locates Visual Studio instance that meets the minimal requirements specified by tools.vs object in global.json.
#
# The following properties of tools.vs are recognized:
#   "version": "{major}.{minor}"
#       Two part minimal VS version, e.g. "15.9", "16.0", etc.
#   "components": ["componentId1", "componentId2", ...]
#       Array of ids of workload components that must be available in the VS instance.
#       See e.g. https://docs.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-enterprise?view=vs-2017
#
# Returns JSON describing the located VS instance (same format as returned by vswhere),
# or $null if no instance meeting the requirements is found on the machine.
#
function LocateVisualStudio([object]$vsRequirements = $null){
  if (-not (IsWindowsPlatform)) {
    throw "Cannot run vswhere on non-Windows platforms."
  }

  if (Get-Member -InputObject $GlobalJson.tools -Name 'vswhere') {
    $vswhereVersion = $GlobalJson.tools.vswhere
  } else {
    # keep this in sync with the VSWhereVersion in DefaultVersions.props
    $vswhereVersion = '3.1.7'
  }

  $vsWhereDir = Join-Path $ToolsDir "vswhere\$vswhereVersion"
  $vsWhereExe = Join-Path $vsWhereDir 'vswhere.exe'

  if (!(Test-Path $vsWhereExe)) {
    Create-Directory $vsWhereDir
    Write-Host "Downloading vswhere $vswhereVersion"
    $ProgressPreference = 'SilentlyContinue' # Don't display the console progress UI - it's a huge perf hit
    Retry({
      Invoke-WebRequest "https://netcorenativeassets.blob.core.windows.net/resource-packages/external/windows/vswhere/$vswhereVersion/vswhere.exe" -UseBasicParsing -OutFile $vswhereExe
    })
  }

  if (!$vsRequirements) {
    if (Get-Member -InputObject $GlobalJson.tools -Name 'vs' -ErrorAction SilentlyContinue) {
      $vsRequirements = $GlobalJson.tools.vs
    } else {
      $vsRequirements = $null
    }
  }

  $args = @('-latest', '-format', 'json', '-requires', 'Microsoft.Component.MSBuild', '-products', '*')

  if (!$excludePrereleaseVS) {
    $args += '-prerelease'
  }

  if ($vsRequirements -and (Get-Member -InputObject $vsRequirements -Name 'version' -ErrorAction SilentlyContinue)) {
    $args += '-version'
    $args += $vsRequirements.version
  }

  if ($vsRequirements -and (Get-Member -InputObject $vsRequirements -Name 'components' -ErrorAction SilentlyContinue)) {
    foreach ($component in $vsRequirements.components) {
      $args += '-requires'
      $args += $component
    }
  }

  $vsInfo =& $vsWhereExe $args | ConvertFrom-Json

  if ($lastExitCode -ne 0) {
    return $null
  }

  # use first matching instance
  return $vsInfo[0]
}

function InitializeBuildTool() {
  if (Test-Path variable:global:_BuildTool) {
    # If the requested msbuild parameters do not match, clear the cached variables.
    if($global:_BuildTool.Contains('ExcludePrereleaseVS') -and $global:_BuildTool.ExcludePrereleaseVS -ne $excludePrereleaseVS) {
      Remove-Item variable:global:_BuildTool
      Remove-Item variable:global:_MSBuildExe
    } else {
      return $global:_BuildTool
    }
  }

  if (-not $msbuildEngine) {
    $msbuildEngine = GetDefaultMSBuildEngine
  }

  # Initialize dotnet cli if listed in 'tools'
  $dotnetRoot = $null
  if (Get-Member -InputObject $GlobalJson.tools -Name 'dotnet') {
    $dotnetRoot = InitializeDotNetCli -install:$restore
  }

  if ($msbuildEngine -eq 'dotnet') {
    if (!$dotnetRoot) {
      Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "/global.json must specify 'tools.dotnet'."
      ExitWithExitCode 1
    }
    $dotnetPath = Join-Path $dotnetRoot (GetExecutableFileName 'dotnet')

    $buildTool = @{ Path = $dotnetPath; Command = 'msbuild'; Tool = 'dotnet'; Framework = 'net' }
  } elseif ($msbuildEngine -eq "vs") {
    try {
      $msbuildPath = InitializeVisualStudioMSBuild -install:$restore
    } catch {
      Write-PipelineTelemetryError -Category 'InitializeToolset' -Message $_
      ExitWithExitCode 1
    }

    $buildTool = @{ Path = $msbuildPath; Command = ""; Tool = "vs"; Framework = "netframework"; ExcludePrereleaseVS = $excludePrereleaseVS }
  } else {
    Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "Unexpected value of -msbuildEngine: '$msbuildEngine'."
    ExitWithExitCode 1
  }

  return $global:_BuildTool = $buildTool
}

function GetDefaultMSBuildEngine() {
  # Presence of tools.vs indicates the repo needs to build using VS msbuild on Windows.
  if (Get-Member -InputObject $GlobalJson.tools -Name 'vs') {
    return 'vs'
  }

  if (Get-Member -InputObject $GlobalJson.tools -Name 'dotnet') {
    return 'dotnet'
  }

  Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "-msbuildEngine must be specified, or /global.json must specify 'tools.dotnet' or 'tools.vs'."
  ExitWithExitCode 1
}

function GetNuGetPackageCachePath() {
  if ($env:NUGET_PACKAGES -eq $null) {
    # Use local cache on CI to ensure deterministic build.
    # Avoid using the http cache as workaround for https://github.com/NuGet/Home/issues/3116
    # use global cache in dev builds to avoid cost of downloading packages.
    # For directory normalization, see also: https://github.com/NuGet/Home/issues/7968
    if ($useGlobalNuGetCache) {
      $env:NUGET_PACKAGES = Join-Path $env:UserProfile '.nuget\packages\'
    } else {
      $env:NUGET_PACKAGES = Join-Path $RepoRoot '.packages\'
    }
  }

  return $env:NUGET_PACKAGES
}

# Returns a full path to an Arcade SDK task project file.
function GetSdkTaskProject([string]$taskName) {
  return Join-Path (Split-Path (InitializeToolset) -Parent) "SdkTasks\$taskName.proj"
}

function InitializeNativeTools() {
  if (-Not (Test-Path variable:DisableNativeToolsetInstalls) -And (Get-Member -InputObject $GlobalJson -Name "native-tools")) {
    $nativeArgs= @{}
    if ($ci) {
      $nativeArgs = @{
        InstallDirectory = "$ToolsDir"
      }
    }
    if ($env:NativeToolsOnMachine) {
      Write-Host "Variable NativeToolsOnMachine detected, enabling native tool path promotion..."
      $nativeArgs += @{ PathPromotion = $true }
    }
    & "$PSScriptRoot/init-tools-native.ps1" @nativeArgs
  }
}

function Read-ArcadeSdkVersion() {
  return $GlobalJson.'msbuild-sdks'.'Microsoft.DotNet.Arcade.Sdk'
}

function InitializeToolset() {
  # For Unified Build/Source-build support, check whether the environment variable is
  # set. If it is, then use this as the toolset build project.
  if ($env:_InitializeToolset -ne $null) {
    return $global:_InitializeToolset = $env:_InitializeToolset
  }

  if (Test-Path variable:global:_InitializeToolset) {
    return $global:_InitializeToolset
  }

  $nugetCache = GetNuGetPackageCachePath

  $toolsetVersion = Read-ArcadeSdkVersion
  $toolsetLocationFile = Join-Path $ToolsetDir "$toolsetVersion.txt"

  if (Test-Path $toolsetLocationFile) {
    $path = Get-Content $toolsetLocationFile -TotalCount 1
    if (Test-Path $path) {
      return $global:_InitializeToolset = $path
    }
  }

  if (-not $restore) {
    Write-PipelineTelemetryError -Category 'InitializeToolset' -Message "Toolset version $toolsetVersion has not been restored."
    ExitWithExitCode 1
  }

  $buildTool = InitializeBuildTool

  $proj = Join-Path $ToolsetDir 'restore.proj'
  $bl = if ($binaryLog) { '/bl:' + (Join-Path $LogDir 'ToolsetRestore.binlog') } else { '' }

  '<Project Sdk="Microsoft.DotNet.Arcade.Sdk"/>' | Set-Content $proj

  MSBuild-Core $proj $bl /t:__WriteToolsetLocation /clp:ErrorsOnly`;NoSummary /p:__ToolsetLocationOutputFile=$toolsetLocationFile

  $path = Get-Content $toolsetLocationFile -Encoding UTF8 -TotalCount 1
  if (!(Test-Path $path)) {
    throw "Invalid toolset path: $path"
  }

  return $global:_InitializeToolset = $path
}

function ExitWithExitCode([int] $exitCode) {
  if ($ci -and $prepareMachine) {
    Stop-Processes
  }
  exit $exitCode
}

# Check if $LASTEXITCODE is a nonzero exit code (NZEC). If so, print a Azure Pipeline error for
# diagnostics, then exit the script with the $LASTEXITCODE.
function Exit-IfNZEC([string] $category = "General") {
  Write-Host "Exit code $LASTEXITCODE"
  if ($LASTEXITCODE -ne 0) {
    $message = "Last command failed with exit code $LASTEXITCODE."
    Write-PipelineTelemetryError -Force -Category $category -Message $message
    ExitWithExitCode $LASTEXITCODE
  }
}

function Stop-Processes() {
  Write-Host 'Killing running build processes...'
  foreach ($processName in $processesToStopOnExit) {
    Get-Process -Name $processName -ErrorAction SilentlyContinue | Stop-Process
  }
}

#
# Executes msbuild (or 'dotnet msbuild') with arguments passed to the function.
# The arguments are automatically quoted.
# Terminates the script if the build fails.
#
function MSBuild() {
  if ($pipelinesLog) {
    $buildTool = InitializeBuildTool

    if ($ci -and $buildTool.Tool -eq 'dotnet') {
      $env:NUGET_PLUGIN_HANDSHAKE_TIMEOUT_IN_SECONDS = 20
      $env:NUGET_PLUGIN_REQUEST_TIMEOUT_IN_SECONDS = 20
      Write-PipelineSetVariable -Name 'NUGET_PLUGIN_HANDSHAKE_TIMEOUT_IN_SECONDS' -Value '20'
      Write-PipelineSetVariable -Name 'NUGET_PLUGIN_REQUEST_TIMEOUT_IN_SECONDS' -Value '20'
    }

    Enable-Nuget-EnhancedRetry

    $toolsetBuildProject = InitializeToolset
    $basePath = Split-Path -parent $toolsetBuildProject
    $selectedPath = Join-Path $basePath (Join-Path $buildTool.Framework 'Microsoft.DotNet.ArcadeLogging.dll')

    if (-not $selectedPath) {
      Write-PipelineTelemetryError -Category 'Build' -Message "Unable to find arcade sdk logger assembly: $selectedPath"
      ExitWithExitCode 1
    }

    $args += "/logger:$selectedPath"
  }

  MSBuild-Core @args
}

#
# Executes msbuild (or 'dotnet msbuild') with arguments passed to the function.
# The arguments are automatically quoted.
# Terminates the script if the build fails.
#
function MSBuild-Core() {
  if ($ci) {
    if (!$binaryLog -and !$excludeCIBinarylog) {
      Write-PipelineTelemetryError -Category 'Build' -Message 'Binary log must be enabled in CI build, or explicitly opted-out from with the -excludeCIBinarylog switch.'
      ExitWithExitCode 1
    }

    if ($nodeReuse) {
      Write-PipelineTelemetryError -Category 'Build' -Message 'Node reuse must be disabled in CI build.'
      ExitWithExitCode 1
    }
  }

  Enable-Nuget-EnhancedRetry

  $buildTool = InitializeBuildTool

  $cmdArgs = "$($buildTool.Command) /m /nologo /clp:Summary /v:$verbosity /nr:$nodeReuse /p:ContinuousIntegrationBuild=$ci"

  if ($warnAsError) {
    $cmdArgs += ' /warnaserror /p:TreatWarningsAsErrors=true'
  }
  else {
    $cmdArgs += ' /p:TreatWarningsAsErrors=false'
  }

  foreach ($arg in $args) {
    if ($null -ne $arg -and $arg.Trim() -ne "") {
      if ($arg.EndsWith('\')) {
        $arg = $arg + "\"
      }
      $cmdArgs += " `"$arg`""
    }
  }

  # Be sure quote the path in case there are spaces in the dotnet installation location.
  $env:ARCADE_BUILD_TOOL_COMMAND = "`"$($buildTool.Path)`" $cmdArgs"

  $exitCode = Exec-Process $buildTool.Path $cmdArgs

  if ($exitCode -ne 0) {
    # We should not Write-PipelineTaskError here because that message shows up in the build summary
    # The build already logged an error, that's the reason it failed. Producing an error here only adds noise.
    Write-Host "Build failed with exit code $exitCode. Check errors above." -ForegroundColor Red

    $buildLog = GetMSBuildBinaryLogCommandLineArgument $args
    if ($null -ne $buildLog) {
      Write-Host "See log: $buildLog" -ForegroundColor DarkGray
    }

    # When running on Azure Pipelines, override the returned exit code to avoid double logging.
    # Skip this when the build is a child of the VMR build.
    if ($ci -and $env:SYSTEM_TEAMPROJECT -ne $null -and !$fromVMR) {
      Write-PipelineSetResult -Result "Failed" -Message "msbuild execution failed."
      # Exiting with an exit code causes the azure pipelines task to log yet another "noise" error
      # The above Write-PipelineSetResult will cause the task to be marked as failure without adding yet another error
      ExitWithExitCode 0
    } else {
      ExitWithExitCode $exitCode
    }
  }
}

function GetMSBuildBinaryLogCommandLineArgument($arguments) {
  foreach ($argument in $arguments) {
    if ($argument -ne $null) {
      $arg = $argument.Trim()
      if ($arg.StartsWith('/bl:', "OrdinalIgnoreCase")) {
        return $arg.Substring('/bl:'.Length)
      }

      if ($arg.StartsWith('/binaryLogger:', 'OrdinalIgnoreCase')) {
        return $arg.Substring('/binaryLogger:'.Length)
      }
    }
  }

  return $null
}

function GetExecutableFileName($baseName) {
  if (IsWindowsPlatform) {
    return "$baseName.exe"
  }
  else {
    return $baseName
  }
}

function IsWindowsPlatform() {
  return [environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
}

function Get-Darc($version) {
  $darcPath  = "$TempDir\darc\$([guid]::NewGuid())"
  if ($version -ne $null) {
    & $PSScriptRoot\darc-init.ps1 -toolpath $darcPath -darcVersion $version | Out-Host
  } else {
    & $PSScriptRoot\darc-init.ps1 -toolpath $darcPath | Out-Host
  }
  return "$darcPath\darc.exe"
}

. $PSScriptRoot\pipeline-logging-functions.ps1

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\')
$EngRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$ArtifactsDir = Join-Path $RepoRoot 'artifacts'
$ToolsetDir = Join-Path $ArtifactsDir 'toolset'
$ToolsDir = Join-Path $RepoRoot '.tools'
$LogDir = Join-Path (Join-Path $ArtifactsDir 'log') $configuration
$TempDir = Join-Path (Join-Path $ArtifactsDir 'tmp') $configuration
$GlobalJson = Get-Content -Raw -Path (Join-Path $RepoRoot 'global.json') | ConvertFrom-Json
# true if global.json contains a "runtimes" section
$globalJsonHasRuntimes = if ($GlobalJson.tools.PSObject.Properties.Name -Match 'runtimes') { $true } else { $false }

Create-Directory $ToolsetDir
Create-Directory $TempDir
Create-Directory $LogDir

Write-PipelineSetVariable -Name 'Artifacts' -Value $ArtifactsDir
Write-PipelineSetVariable -Name 'Artifacts.Toolset' -Value $ToolsetDir
Write-PipelineSetVariable -Name 'Artifacts.Log' -Value $LogDir
Write-PipelineSetVariable -Name 'TEMP' -Value $TempDir
Write-PipelineSetVariable -Name 'TMP' -Value $TempDir

# Import custom tools configuration, if present in the repo.
# Note: Import in global scope so that the script set top-level variables without qualification.
if (!$disableConfigureToolsetImport) {
  $configureToolsetScript = Join-Path $EngRoot 'configure-toolset.ps1'
  if (Test-Path $configureToolsetScript) {
    . $configureToolsetScript
    if ((Test-Path variable:failOnConfigureToolsetError) -And $failOnConfigureToolsetError) {
      if ((Test-Path variable:LastExitCode) -And ($LastExitCode -ne 0)) {
        Write-PipelineTelemetryError -Category 'Build' -Message 'configure-toolset.ps1 returned a non-zero exit code'
        ExitWithExitCode $LastExitCode
      }
    }
  }
}

#
# If $ci flag is set, turn on (and log that we did) special environment variables for improved Nuget client retry logic.
#
function Enable-Nuget-EnhancedRetry() {
    if ($ci) {
      Write-Host "Setting NUGET enhanced retry environment variables"
      $env:NUGET_ENABLE_ENHANCED_HTTP_RETRY = 'true'
      $env:NUGET_ENHANCED_MAX_NETWORK_TRY_COUNT = 6
      $env:NUGET_ENHANCED_NETWORK_RETRY_DELAY_MILLISECONDS = 1000
      $env:NUGET_RETRY_HTTP_429 = 'true'
      Write-PipelineSetVariable -Name 'NUGET_ENABLE_ENHANCED_HTTP_RETRY' -Value 'true'
      Write-PipelineSetVariable -Name 'NUGET_ENHANCED_MAX_NETWORK_TRY_COUNT' -Value '6'
      Write-PipelineSetVariable -Name 'NUGET_ENHANCED_NETWORK_RETRY_DELAY_MILLISECONDS' -Value '1000'
      Write-PipelineSetVariable -Name 'NUGET_RETRY_HTTP_429' -Value 'true'
    }
}
