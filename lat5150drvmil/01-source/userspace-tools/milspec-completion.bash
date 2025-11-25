# Bash completion for milspec-control and milspec-monitor
# Install to /etc/bash_completion.d/milspec-completion

_milspec_control()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    local commands="status mode5 dsmil activate measure wipe"
    
    # Options
    local options="-d --device -f --force -c --color -v --verbose -h --help"
    
    # Mode 5 levels
    local mode5_levels="0 1 2 3 4"
    
    # DSMIL modes
    local dsmil_modes="0 1 2 3"
    
    case "${prev}" in
        milspec-control)
            COMPREPLY=( $(compgen -W "${commands} ${options}" -- ${cur}) )
            return 0
            ;;
        mode5)
            COMPREPLY=( $(compgen -W "${mode5_levels}" -- ${cur}) )
            return 0
            ;;
        dsmil)
            COMPREPLY=( $(compgen -W "${dsmil_modes}" -- ${cur}) )
            return 0
            ;;
        -d|--device)
            # Suggest /dev/milspec
            COMPREPLY=( $(compgen -W "/dev/milspec" -- ${cur}) )
            return 0
            ;;
        *)
            # Check if we're completing after a command
            local i cmd_found=0
            for (( i=1; i < ${#COMP_WORDS[@]}-1; i++ )); do
                if [[ " ${commands} " =~ " ${COMP_WORDS[i]} " ]]; then
                    cmd_found=1
                    break
                fi
            done
            
            if [ $cmd_found -eq 0 ]; then
                # No command yet, offer commands and options
                COMPREPLY=( $(compgen -W "${commands} ${options}" -- ${cur}) )
            else
                # Command already given, only offer options
                COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
            fi
            ;;
    esac
}

_milspec_monitor()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Options
    local options="-d --device -m --mode -s --status -t --timestamp -c --color -l --log -v --verbose -h --help"
    
    # Monitor modes
    local modes="sysfs chardev both"
    
    case "${prev}" in
        -d|--device)
            COMPREPLY=( $(compgen -W "/dev/milspec" -- ${cur}) )
            return 0
            ;;
        -m|--mode)
            COMPREPLY=( $(compgen -W "${modes}" -- ${cur}) )
            return 0
            ;;
        -l|--log)
            # File completion
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
            ;;
    esac
}

# Register completions
complete -F _milspec_control milspec-control
complete -F _milspec_monitor milspec-monitor
