#!/bin/bash
#
# CVE Scraper Service Manager
# Run Telegram CVE scraper as background service
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PID_FILE="rag_system/cve_scraper.pid"
LOG_FILE="rag_system/cve_scraper.log"
PYTHON="python3"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "❌ CVE scraper already running (PID: $PID)"
                exit 1
            fi
        fi

        echo "Starting CVE scraper..."
        nohup $PYTHON rag_system/telegram_cve_scraper.py > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "✓ CVE scraper started (PID: $(cat $PID_FILE))"
        echo "  Log: $LOG_FILE"
        ;;

    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "❌ CVE scraper not running"
            exit 1
        fi

        PID=$(cat "$PID_FILE")
        echo "Stopping CVE scraper (PID: $PID)..."
        kill $PID
        rm "$PID_FILE"
        echo "✓ CVE scraper stopped"
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start
        ;;

    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "✓ CVE scraper running (PID: $PID)"
                echo
                $PYTHON rag_system/telegram_cve_scraper.py --stats
            else
                echo "❌ CVE scraper not running (stale PID file)"
                rm "$PID_FILE"
            fi
        else
            echo "❌ CVE scraper not running"
        fi
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found"
        fi
        ;;

    update)
        echo "Forcing RAG update with all CVEs..."
        $PYTHON rag_system/telegram_cve_scraper.py --update-rag
        ;;

    *)
        echo "CVE Scraper Service Manager"
        echo
        echo "Usage: $0 {start|stop|restart|status|logs|update}"
        echo
        echo "Commands:"
        echo "  start   - Start CVE scraper in background"
        echo "  stop    - Stop CVE scraper"
        echo "  restart - Restart CVE scraper"
        echo "  status  - Check status and show statistics"
        echo "  logs    - Show live logs (tail -f)"
        echo "  update  - Force RAG embedding update"
        exit 1
        ;;
esac
