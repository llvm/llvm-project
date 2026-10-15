import subprocess


def attach(pid: int, suspended: bool = False) -> None:
    script = """
	    on run argv
	        set targetPID to item 1 of argv as integer
	        set shouldSuspend to item 2 of argv as boolean

	        tell application "Xcode"
	            activate

	            if (count of workspace documents) is greater than 0 then
	                set debuggingWorkspace to workspace document 1
	            else
	                set debuggingWorkspace to create temporary debugging workspace
	            end if

	            attach debuggingWorkspace to process identifier targetPID suspended shouldSuspend
	        end tell
	    end run
    """

    subprocess.run(["osascript", "-e", script, str(pid), str(suspended)], check=True)
